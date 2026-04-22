import os
import sys
import cv2
import time
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# YOLO imports
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from OneEuroFilter import OneEuroFilter

# Physformer imports
from physformer import PhysFormerX, bpm_from_signal

# =======================================================
# 1. YOLO Body-Part Extraction Helper Functions
# =======================================================
def create_filter(fps):
    return [
        OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
        OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
        OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
        OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007)
    ]

def crop_and_resize(frame, box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(crop, (w, h))

def run_yolo_tracking(video_path, output_dir, yolo_weights):
    print(">>> [1/2] Running YOLOv8 specific body-part extraction...")
    model = YOLO(yolo_weights)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # fallback
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_nose_path  = os.path.join(output_dir, "nose.mp4")
    out_face_path  = os.path.join(output_dir, "face.mp4")
    out_chest_path = os.path.join(output_dir, "chest.mp4")
    
    # Store dimensions to keep output matching
    out_nose  = cv2.VideoWriter(out_nose_path, fourcc, fps, (w, h))
    out_face  = cv2.VideoWriter(out_face_path, fourcc, fps, (w, h))
    out_chest = cv2.VideoWriter(out_chest_path, fourcc, fps, (w, h))
    
    CLASS_MAP = {0: "face", 1: "nose", 2: "chest"}
    filters = {
        "nose": create_filter(fps),
        "face": create_filter(fps),
        "chest": create_filter(fps)
    }
    
    pbar = tqdm(total=total_frames, desc="YOLO Tracking", unit="frame")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        current_time = frame_idx / fps
        frame_idx += 1
        
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.25, verbose=False, imgsz=320,
            device=0 if torch.cuda.is_available() else 'cpu'
        )[0]
        
        outputs = {
            "nose": np.zeros((h, w, 3), dtype=np.uint8),
            "face": np.zeros((h, w, 3), dtype=np.uint8),
            "chest": np.zeros((h, w, 3), dtype=np.uint8)
        }
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            seen_classes = set()
            for box, cls in zip(boxes, classes):
                cls_name = CLASS_MAP.get(cls, None)
                if cls_name is None or cls_name in seen_classes: continue
                seen_classes.add(cls_name)
                
                f = filters[cls_name]
                x1 = f[0](box[0], current_time)
                y1 = f[1](box[1], current_time)
                x2 = f[2](box[2], current_time)
                y2 = f[3](box[3], current_time)
                smooth_box = np.array([x1, y1, x2, y2])
                
                outputs[cls_name] = crop_and_resize(frame, smooth_box, w, h)
                
        out_nose.write(outputs["nose"])
        out_face.write(outputs["face"])
        out_chest.write(outputs["chest"])
        
    cap.release()
    out_nose.release()
    out_face.release()
    out_chest.release()
    pbar.close()
    
    print(f"Saved crops to {output_dir}")
    return out_face_path, fps


# =======================================================
# 2. Physformer rPPG Prediction Helper Functions
# =======================================================
def run_physformer_inference(face_video_path, weights_path, output_dir, original_fps):
    print("\n>>> [2/2] Running Physformer Inference on extracted face video...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(weights_path):
        print(f"⚠️ Warning: Physformer weights not found at '{weights_path}'.")
        print("Model predicting zeros (or skipping). Please provide correct weights using --physformer_weights")
        return None
        
    # Load model
    model = PhysFormerX(seq_len=300).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    # Process Video (extract 64x64 spatial representation)
    cap = cv2.VideoCapture(face_video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = original_fps
    
    print("Extracting spatial frames for Physformer...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("No face tracking frames available. Returning.")
        return None
        
    # Prepare Input Tensor (6 channels: standard + differences)
    # [T, H, W, C] -> [T, H, W, 3] -> [3, T, H, W]
    vid = torch.tensor(np.array(frames)).float() / 255.0
    vid = vid.permute(3, 0, 1, 2)
    
    diff = torch.zeros_like(vid)
    diff[:, 1:] = vid[:, 1:] - vid[:, :-1]
    
    vid_6ch = torch.cat([vid, diff], dim=0).unsqueeze(0).to(device) # [1, 6, T, H, W]
    
    # Run prediction block
    print("Running forward pass (PhysFormerX)...")
    with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        pred_signal = model(vid_6ch).squeeze().float().cpu().numpy() # Expected output [T]
    
    if pred_signal.ndim == 0:
        pred_signal = np.expand_dims(pred_signal, axis=0)

    # Post processing output signal
    pred_bpm = bpm_from_signal(pred_signal[None, :], int(fps))[0]
    print(f"🌟 Calculated Heart Rate: {pred_bpm:.2f} BPM")
    
    # Visualization & Savings
    plt.figure(figsize=(12, 4))
    plt.plot(pred_signal, label=f"Predicted Signal (BPM: {pred_bpm:.2f})")
    plt.title("Physformer Predicted rPPG Signal")
    plt.xlabel("Frames")
    plt.ylabel("Amplitude")
    plt.legend()
    plot_path = os.path.join(output_dir, "rppg_waveform.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        "bpm": float(pred_bpm),
        "signal": pred_signal.tolist()
    }
    
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results successfully saved into {output_dir}")
    return results

# =======================================================
# 3. Execution Main
# =======================================================
def main():
    parser = argparse.ArgumentParser("YOLO + Physformer Inference Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input raw video file")
    parser.add_argument("--yolo_weights", type=str, default="models/yolov8/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--physformer_weights", type=str, default="models/physformer.pt", help="Path to trained PhysFormerX weights")
    parser.add_argument("--output_base", type=str, default="outputs", help="Base directory for runtime outputs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file {args.video} not found on the system.")
        return
        
    # Setup runtime subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = os.path.join(args.output_base, timestamp)
    os.makedirs(runtime_dir, exist_ok=True)
    
    print(f"\n🚀 Pipeline initializing... Results will be stored in: {runtime_dir}\n")
    
    # 1. Run inference block for YOLO object tracking
    face_video_path, fps = run_yolo_tracking(args.video, runtime_dir, args.yolo_weights)
    
    # 2. Extract rPPG with Physformer model using the targeted "face" features
    run_physformer_inference(face_video_path, args.physformer_weights, runtime_dir, fps)

if __name__ == "__main__":
    main()
