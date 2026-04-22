"""
Advanced rPPG Preprocessor for UBFC-rPPG
─────────────────────────────────────────
Improvements over the original:
  1. Face-box temporal smoothing (EMA) to reduce jitter
  2. YCbCr skin-region masking — zeros out eyes, hair, background
  3. Butterworth bandpass (0.7–2.5 Hz) on ground-truth PPG
  4. Per-frame temporal normalization (running-mean subtraction)
  5. Saves subject_id in each .pt for subject-level train/test splits
  6. Configurable for both local and Kaggle paths

Usage (local):
  python preprocessor.py --video_dir "E:/CS776 Project/Dataset"
                         --yolo_weights "E:/CS776 Project/best.pt"
                         --output_dir "E:/CS776 Project/preprocessed_physformer"
"""

import os, glob, re, argparse
import cv2
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SEQ_LENGTH = 300        # frames per clip (longer = better HR estimation)
FACE_SIZE  = 128        # spatial resolution
STRIDE     = 150        # stride between chunks (50% overlap)
BP_LO      = 0.7        # bandpass low  (Hz) → ~42 BPM
BP_HI      = 2.5        # bandpass high (Hz) → ~150 BPM
BP_ORDER   = 2          # Butterworth order
EMA_ALPHA  = 0.7        # face-box smoothing strength (higher = more smoothing)

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def bandpass_filter(signal, fps, lo=BP_LO, hi=BP_HI, order=BP_ORDER):
    """Apply zero-phase Butterworth bandpass to 1-D signal."""
    nyq = fps / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    # Clamp to valid range
    lo_n = max(lo_n, 0.001)
    hi_n = min(hi_n, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, signal).astype(np.float32)


def skin_mask_ycrcb(frame_rgb):
    """
    Binary skin mask via YCbCr thresholding.
    Keeps forehead/cheeks, removes eyes, hair, background.
    """
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    # Standard skin-color ranges in YCbCr
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask  # uint8, 0 or 255


def extract_subject_id(vid_filename):
    """Extract subject identifier from UBFC filename."""
    name = os.path.splitext(vid_filename)[0]
    # Dataset 2: vid_2_sub<N>
    m = re.search(r'sub(\d+)', name)
    if m:
        return f"sub{m.group(1)}"
    # Dataset 1: vid_1_<N>gt  or vid_1_after_exercise
    m = re.search(r'vid_1_(\w+)', name)
    if m:
        return f"ds1_{m.group(1)}"
    return name


def smooth_box(prev_box, new_box, alpha=EMA_ALPHA):
    """Exponential moving average of bounding box [x1,y1,x2,y2]."""
    if prev_box is None:
        return new_box
    return [alpha * p + (1 - alpha) * n for p, n in zip(prev_box, new_box)]


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def preprocess_dataset(video_dir, yolo_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO
    yolo = YOLO(yolo_path, task='detect')

    # Discover videos
    video_paths = sorted(glob.glob(os.path.join(video_dir, '**/*.avi'), recursive=True))
    print(f"Found {len(video_paths)} videos in {video_dir}")

    window_count = 0

    for vid_path in video_paths:
        vid_file = os.path.basename(vid_path)
        vid_dir_local = os.path.dirname(vid_path)
        subject_id = extract_subject_id(vid_file)

        # ── Match ground-truth file ─────────────────────────
        if "vid_1_" in vid_file:
            # Dataset 1: gtdump format (CSV: timestamp, HR, SpO2, PPG)
            gt_name = vid_file.replace("vid_1_", "ground_truth_1_").replace(".avi", ".txt")
            gt_format = "ds1"
        else:
            # Dataset 2: space-separated (line1=PPG, line2=HR, line3=time)
            gt_name = vid_file.replace("vid_2_", "ground_truth_2_").replace(".avi", ".txt")
            gt_format = "ds2"

        gt_path = os.path.join(vid_dir_local, gt_name)
        if not os.path.exists(gt_path):
            print(f"  SKIP {vid_file}: GT not found ({gt_name})")
            continue

        print(f"Processing: {vid_file}  [subject={subject_id}]")

        # ── Read video with face detection + smoothing ───────
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        prev_box = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO face detection
            results = yolo(frame, classes=[0], verbose=False)

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy().tolist()
                h_f, w_f = frame.shape[:2]
                box = [max(0, box[0]), max(0, box[1]),
                       min(w_f, box[2]), min(h_f, box[3])]
                # Smooth bounding box
                prev_box = smooth_box(prev_box, box)
                x1, y1, x2, y2 = [int(c) for c in prev_box]
                face = frame[y1:y2, x1:x2]
            elif prev_box is not None:
                # Use last known box if detection fails
                x1, y1, x2, y2 = [int(c) for c in prev_box]
                face = frame[y1:y2, x1:x2]
            else:
                face = frame

            if face.size == 0:
                face = frame

            # Resize & convert to RGB
            face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Apply skin mask — zero out non-skin pixels
            mask = skin_mask_ycrcb(face_rgb)
            mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
            face_masked = face_rgb * mask_3ch

            frames.append(face_masked)

        cap.release()
        frames_array = np.array(frames, dtype=np.uint8)

        if len(frames_array) < SEQ_LENGTH:
            print(f"  SKIP: too short ({len(frames_array)} < {SEQ_LENGTH})")
            continue

        # ── Temporal normalization of video ───────────────────
        # Subtract running mean to remove slow lighting drift
        frames_float = frames_array.astype(np.float32)
        # Compute per-frame mean intensity
        frame_means = frames_float.mean(axis=(1, 2, 3), keepdims=True)
        # Running average with window = 30 frames
        from scipy.ndimage import uniform_filter1d
        smoothed_means = uniform_filter1d(frame_means.squeeze(), size=30, axis=0)
        smoothed_means = smoothed_means.reshape(-1, 1, 1, 1)
        # Normalize: subtract slow drift, add global mean back
        global_mean = frame_means.mean()
        frames_float = frames_float - smoothed_means + global_mean
        frames_float = np.clip(frames_float, 0, 255)
        frames_norm = frames_float.astype(np.uint8)

        # ── Read & process ground truth ───────────────────────
        try:
            if gt_format == "ds1":
                # CSV: timestamp_ms, HR, SpO2, PPG_value
                data = np.loadtxt(gt_path, delimiter=',')
                gt_ppg = data[:, 3].astype(np.float64)
                gt_time = (data[:, 0] / 1000.0).astype(np.float64)
            else:
                # Space-separated: line1=PPG, line2=HR, line3=timestamps
                with open(gt_path, 'r') as f:
                    lines = f.readlines()
                gt_ppg = np.array([float(x) for x in lines[0].strip().split()], dtype=np.float64)
                gt_time = np.array([float(x) for x in lines[2].strip().split()], dtype=np.float64)

            # Zero-offset time
            gt_time = gt_time - gt_time[0]

            # Remove duplicate timestamps
            _, unique_idx = np.unique(gt_time, return_index=True)
            gt_time = gt_time[unique_idx]
            gt_ppg = gt_ppg[unique_idx]

            # Interpolate to video frame timestamps
            video_timestamps = np.linspace(0, (len(frames_norm) - 1) / fps, len(frames_norm))
            interp_func = interp1d(gt_time, gt_ppg, kind='cubic', fill_value='extrapolate')
            aligned_ppg = interp_func(video_timestamps).astype(np.float64)

            # Bandpass filter the GT PPG signal
            aligned_ppg = bandpass_filter(aligned_ppg, fps)

        except Exception as e:
            print(f"  SKIP: GT processing error: {e}")
            continue

        # ── Chunk and save ────────────────────────────────────
        for start in range(0, len(frames_norm) - SEQ_LENGTH + 1, STRIDE):
            win_frames = frames_norm[start:start + SEQ_LENGTH]
            win_ppg = aligned_ppg[start:start + SEQ_LENGTH]

            # Z-normalize PPG per chunk
            mu, std = win_ppg.mean(), win_ppg.std()
            if std < 1e-6:
                continue  # skip dead signal
            win_ppg = (win_ppg - mu) / std

            # Save as [C, T, H, W] uint8 video + float32 ppg + subject_id
            video_tensor = torch.tensor(win_frames, dtype=torch.uint8).permute(3, 0, 1, 2)
            ppg_tensor = torch.tensor(win_ppg, dtype=torch.float32)

            save_path = os.path.join(output_dir, f"window_{window_count}.pt")
            torch.save({
                'video': video_tensor,
                'ppg': ppg_tensor,
                'subject_id': subject_id,
                'fps': fps
            }, save_path)
            window_count += 1

        print(f"  ✔ {window_count} total windows saved so far")

    print(f"\n{'='*60}")
    print(f"Preprocessing complete! {window_count} chunks → {output_dir}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UBFC-rPPG Preprocessor')
    parser.add_argument('--video_dir', type=str,
                        default='E:/CS776 Project/Dataset',
                        help='Directory containing .avi videos and GT files')
    parser.add_argument('--yolo_weights', type=str,
                        default='E:/CS776 Project/best.pt',
                        help='Path to YOLO face detection weights')
    parser.add_argument('--output_dir', type=str,
                        default='E:/CS776 Project/preprocessed_physformer',
                        help='Output directory for .pt chunks')
    parser.add_argument('--seq_length', type=int, default=SEQ_LENGTH)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--face_size', type=int, default=FACE_SIZE)
    args = parser.parse_args()

    SEQ_LENGTH = args.seq_length
    STRIDE = args.stride
    FACE_SIZE = args.face_size

    preprocess_dataset(args.video_dir, args.yolo_weights, args.output_dir)