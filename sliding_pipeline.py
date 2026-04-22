import os
import cv2
import time
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm

# YOLO & Tracking
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from OneEuroFilter import OneEuroFilter

# Physformer
from physformer import PhysFormerX, bpm_from_signal
from preprocessor import skin_mask_ycrcb
from scipy.ndimage import uniform_filter1d
from scipy.signal import periodogram

def rr_from_signal(signals: np.ndarray, fps: int = 30) -> np.ndarray:
    rrs = []
    for s in signals:
        s = s - np.mean(s)
        freqs, psd = periodogram(s, fs=fps, nfft=2048, detrend='constant', window='hann')
        valid = np.where((freqs >= 0.02) & (freqs <= 0.38))[0] # 12 to 60 BRPM
        if len(valid) > 0:
            rrs.append(freqs[valid][np.argmax(psd[valid])] * 60.0)
        else:
            rrs.append(20.0)
    return np.array(rrs)

def smooth_box(prev_box, new_box, alpha=0.8):
    """Exponential moving average of bounding box [x1,y1,x2,y2] mapping to training properties."""
    if prev_box is None:
        return new_box
    return [alpha * p + (1 - alpha) * n for p, n in zip(prev_box, new_box)]

def main(args_obj=None):
    if args_obj is None:
        parser = argparse.ArgumentParser("Sliding Window Physformer Pipeline")
        parser.add_argument("--video", required=True, help="Path to input video")
        parser.add_argument("--yolo_weights", default="models/yolov8/weights/best.pt", help="YOLO checkpoint local path")
        parser.add_argument("--physformer_weights", default="models/physformer.pt", help="Physformer checkpoint local path")
        parser.add_argument("--hf_repo", default="pushpender-23/yolo-cs776-model", help="Hugging Face repository ID for automated fallback downloads")
        parser.add_argument("--yolo_hf_file", default="train_medium_v1/weights/best.pt", help="YOLO filename in HF repo")
        parser.add_argument("--physformer_hf_file", default="physformer.pt", help="Physformer filename in HF repo")
        parser.add_argument("--output", default="outputs/annotated_output.mp4", help="Path to save annotated MP4")
        parser.add_argument("--window_sec", type=int, default=10, help="Window size in seconds")
        parser.add_argument("--stride_sec", type=int, default=1, help="Stride step in seconds (overlapped processing)")
        args = parser.parse_args()
    else:
        args = args_obj
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------------------------------------
    # 1. Initialize Models (with HF Kaggle Auto-Fetch & OpenVINO CPU Acceleration)
    # -------------------------------------------------------------
    print("⚙️ Initializing YOLOv8 & Physformer...")
    
    yolo_path = args.yolo_weights
    if not os.path.exists(yolo_path):
        print(f"⚠️ YOLO weights not found locally. Auto-fetching from HF: {args.hf_repo}...")
        yolo_path = hf_hub_download(repo_id=args.hf_repo, filename=args.yolo_hf_file, local_dir="models")
        
    # OpenVINO CPU Acceleration for Hugging Face Spaces Zero-GPU
    if not torch.cuda.is_available():
        print("⚡ CPU detected! Optimizing YOLOv8 using OpenVINO for massive speedup...")
        openvino_dir = yolo_path.replace('.pt', '_openvino_model')
        if not os.path.exists(openvino_dir):
            print("Exporting model to OpenVINO format for the first time...")
            base_model = YOLO(yolo_path)
            base_model.export(format='openvino', imgsz=320, dynamic=True)
        yolo_model = YOLO(openvino_dir)
    else:
        yolo_model = YOLO(yolo_path)
    
    physformer_path = args.physformer_weights
    if not os.path.exists(physformer_path):
        print(f"⚠️ Physformer weights not found locally. Auto-fetching from HF: {args.hf_repo}...")
        physformer_path = hf_hub_download(repo_id=args.hf_repo, filename=args.physformer_hf_file, local_dir="models")
        
    vid_model = PhysFormerX(seq_len=300).to(device)
    vid_model.load_state_dict(torch.load(physformer_path, map_location=device, weights_only=True))
    vid_model.eval()

    # -------------------------------------------------------------
    # 2. Extract Global Video Properties
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_frames = args.window_sec * fps
    stride_frames = args.stride_sec * fps

    prev_box = None
    boxes_over_time = [] 
    face_crops = []      
    
    # -------------------------------------------------------------
    # Pass 1: Track Boxes and Cache standard Face Croppings
    # -------------------------------------------------------------
    print(f"\n🚀 Phase 1/3: Reading frames & extracting YOLO face coordinates...")
    frame_idx = 0
    pbar = tqdm(total=total_frames, unit="frames")
    
    batch_size = 16
    frame_batch = []
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_batch.append(frame)
            
        if len(frame_batch) == batch_size or (not ret and len(frame_batch) > 0):
            res_list = yolo_model(frame_batch, classes=[0], verbose=False, imgsz=320, device=0 if torch.cuda.is_available() else 'cpu')
            
            for b_frame, results in zip(frame_batch, res_list):
                box_saved = None
                crop_saved = np.zeros((128, 128, 3), dtype=np.uint8)
                
                if len(results.boxes) > 0:
                    box = results.boxes[0].xyxy[0].cpu().numpy().tolist()
                    box_coords = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
                    prev_box = smooth_box(prev_box, box_coords, alpha=0.8) # High rigidity matching preprocessor
                    box_saved = prev_box
                    x1, y1, x2, y2 = prev_box
                    
                    x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
                    x2_i, y2_i = min(w, int(x2)), min(h, int(y2))
                    crop = b_frame[y1_i:y2_i, x1_i:x2_i]
                    if crop.size > 0:
                        # Spatial Preprocessing: Isolate Skin Only exactly matching preprocessor bounds
                        crop = cv2.resize(crop, (128, 128))
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mask = skin_mask_ycrcb(crop_rgb)
                        mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
                        crop_saved = crop_rgb * mask_3ch
                elif prev_box is not None:
                    # Fallback cleanly to last known box if YOLO blinks out
                    box_saved = prev_box
                    x1, y1, x2, y2 = prev_box
                    x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
                    x2_i, y2_i = min(w, int(x2)), min(h, int(y2))
                    crop = b_frame[y1_i:y2_i, x1_i:x2_i]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (128, 128))
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mask = skin_mask_ycrcb(crop_rgb)
                        mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
                        crop_saved = crop_rgb * mask_3ch
                
                boxes_over_time.append(box_saved)
                face_crops.append(crop_saved)
                frame_idx += 1
                pbar.update(1)
                
            frame_batch = []
            
        if not ret: break
        
    pbar.close()
    cap.release()
    
    # -------------------------------------------------------------
    # Pass 2: Global Normalization & Execute NATIVE Physformer Inference
    # -------------------------------------------------------------
    print(f"\n🚀 Phase 2/3: Applying Native Continuous Sequence Physformer Inference...")
    
    from torchvision.transforms import Resize
    resize_layer = Resize((64, 64), antialias=True)
    
    # ── Temporal Normalization (Over Entire Sequence to avoid Edge shocks) ──
    print("Normalizing global clip drift...")
    frames_float = np.array(face_crops).astype(np.float32)
    frame_means = frames_float.mean(axis=(1, 2, 3), keepdims=True)
    smoothed_means = uniform_filter1d(frame_means.squeeze(), size=30, axis=0) # 30 frames window rolling
    smoothed_means = smoothed_means.reshape(-1, 1, 1, 1)
    global_mean = frame_means.mean()
        
    frames_float = frames_float - smoothed_means + global_mean
    frames_float = np.clip(frames_float, 0, 255)
    
    # Prepare full continuous multi-channel temporal tensor batch dynamically
    vid = torch.tensor(frames_float).float() / 255.0
    vid = vid.permute(3, 0, 1, 2) 
    
    # Native PyTorch Antialiasing matching Physformer dataset pipeline
    vid = resize_layer(vid.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    diff = torch.zeros_like(vid)
    diff[:, 1:] = vid[:, 1:] - vid[:, :-1]
    vid_6ch = torch.cat([vid, diff], dim=0).unsqueeze(0).to(device)
    
    print("Executing complete continuous temporal sequence through Physformer natively...")
    with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        pred_signal = vid_model(vid_6ch).squeeze().float().cpu().numpy()
        
    if pred_signal.ndim == 0:
        pred_signal = np.expand_dims(pred_signal, axis=0)
        
    # Dynamically predict unified average BPM strictly bound to this entire sequence
    final_bpm = bpm_from_signal(pred_signal[None, :], fps)[0]
    final_rr = rr_from_signal(pred_signal[None, :], fps)[0]
    print(f"🌟 Final Native Average BPM calculated: {final_bpm:.1f} BPM")
    print(f"🌟 Final Native Average RR calculated: {final_rr:.1f} BRPM")
    
    # ── Map Dynamic Causal History Frequencies ───────────────────
    print(f"Executing causal sliding FFT over PPG history (Window: {args.window_sec}s, Stride: {args.stride_sec}s)...")
    bpm_per_frame = [0.0] * total_frames
    rr_per_frame  = [0.0] * total_frames
    
    last_bpm, last_rr = 0.0, 0.0
    for i in tqdm(range(window_frames, total_frames), desc="Sliding Causal FFTs"):
        if (i - window_frames) % stride_frames == 0:
            start_idx = i - window_frames
            local_signal = pred_signal[start_idx:i]
            last_bpm = bpm_from_signal(local_signal[None, :], fps)[0]
            last_rr = rr_from_signal(local_signal[None, :], fps)[0]
            
        bpm_per_frame[i] = last_bpm
        rr_per_frame[i] = last_rr
        
    # ── Plot Native Reconstructed Signal ───────────────────
    print(f"\n📈 Generating continuous PPG waveform plot...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    final_signal = pred_signal
    
    plt.figure(figsize=(12, 4))
    time_axis = np.arange(len(final_signal)) / fps
    plt.plot(time_axis, final_signal, color='red', linewidth=1.5)
    plt.title(f"Native Inferred rPPG Waveform (Average: {final_bpm:.1f} BPM)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Inferred Amplitude")
    plt.grid(True, alpha=0.3)
    
    plot_path = args.output.replace('.mp4', '_waveform.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Waveform plot successfully saved to: {plot_path}")
            
    # -------------------------------------------------------------
    # Pass 3: Annotate Video Pipeline Visually
    # -------------------------------------------------------------
    print(f"\n🚀 Phase 3/3: Reconstructing explicit video with bound annotations...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    pbar = tqdm(total=total_frames, unit="frames")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        box = boxes_over_time[frame_idx]
        bpm = bpm_per_frame[frame_idx]
        rr = rr_per_frame[frame_idx]
        
        # Perform visual overlay plotting
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            
            # Thick Green Bounding Face Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if bpm > 0:
                text = f"HR: {bpm:.1f} BPM | RR: {rr:.1f} BRPM"
                color = (0, 255, 0)
            else:
                text = f"Gathering {args.window_sec}s history..."
                color = (0, 165, 255) # Orange
                
            # Draw Top Box Overlay
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Video Processing Successful! Annotated timeline stored at: {args.output}")
    
    # ── Ultra-Fast Web Compatibility Transcode ───────────────────
    import subprocess
    web_ready_path = args.output.replace(".mp4", "_web_h264.mp4")
    print("🎬 Transcoding to Web-Safe H.264 for instant Gradio playback...")
    # Bypass Gradio's notoriously slow synchronous converter and compile ultrafast
    subprocess.run(["ffmpeg", "-y", "-i", args.output, "-vcodec", "libx264", "-preset", "ultrafast", web_ready_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   
    return final_bpm, final_rr, plot_path, web_ready_path

if __name__ == "__main__":
    main()
