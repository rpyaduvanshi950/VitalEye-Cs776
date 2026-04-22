# 🫀 VitalEye: Real-Time Physiological Monitoring Deployment

VitalEye is a state-of-the-art physiological monitoring system that uses computer vision to estimate Heart Rate (BPM) and Respiratory Rate (BRPM) from standard RGB video streams. This directory contains the optimized deployment-ready code used for the **Hugging Face Spaces** web application.

---

## 📂 Project Structure

```text
VitalEye_CS776_Deploy/
├── app.py                # Main Gradio Web Application interface
├── sliding_pipeline.py   # Core Logic: Causal sliding window inference (Continuous)
├── pipeline.py          # Batch Logic: Sequential ROI extraction & inference
├── physformer.py         # PhysFormerX Model Architecture & Signal Processing
├── preprocessor.py       # Skin segmentation & YCbCr spatial masking
├── OneEuroFilter.py      # Smoothing filter for jitter-free tracking
├── Dockerfile           # Containerization for Hugging Face Spaces
├── requirements.txt      # Python dependencies
└── models/               # Local weights storage (Auto-fetched if missing)
    ├── physformer.pt      # PhysFormerX weights
    └── yolov8/
        └── weights/
            └── best.pt   # YOLOv8 ROI tracking weights
```

---

## 🧠 Key Code Components

### 1. `app.py` (The Interface)
The entry point for the web application. It uses **Gradio** to provide a user-friendly UI. It handles:
- Video file uploads and input validation.
- Integration with the `sliding_pipeline`.
- Displaying real-time feedback, annotated videos, and rPPG waveforms.

### 2. `sliding_pipeline.py` (The Engine)
This script implements the **Causal Sliding Window** logic:
- **Windowing**: Processes video in 10-second segments with a 1-second stride.
- **OpenVINO Acceleration**: Automatically detects CPU environments (like HF Spaces) and compiles YOLOv8 to OpenVINO format for a **5x-10x speed boost**.
- **Real-Time Smoothing**: Uses `OneEuroFilter` and `Exponential Moving Average (EMA)` to ensure the YOLO bounding boxes are stable, preventing motion noise in the signal.
- **Continuous FFT**: Performs Fast Fourier Transforms on the sliding signal buffer to provide a dynamic HR/RR estimate.

### 3. `physformer.py` (The Model)
Contains the **PhysFormerX** architecture:
- **3D Residual Backbone**: Extracts spatiotemporal features from the video.
- **SE-Attention**: Dynamically focuses on the color channels (primarily Green) that carry the strongest pulse information.
- **Signal Extraction**: Reduces high-dimensional video data into a 1D physiological waveform.

### 4. `preprocessor.py` (Spatial Engineering)
Handles the critical task of skin isolation.
- **YCbCr Masking**: Filters out non-skin pixels (eyes, background, hair).
- **Difference Frames**: Generates the 6-channel representation ($RGB + \Delta RGB$) which amplifies the pulse signal while neutralizing static features.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+
- FFMPEG installed on your system (for video encoding)

### 2. Setup Environment
```bash
# Clone the repository and navigate to the deploy folder
cd VitalEye_CS776_Deploy

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the App
```bash
python app.py
```
The app will be available at `http://localhost:7860`. The system will automatically download the necessary model weights from the Hugging Face Hub if they are not found in the `models/` directory.

---

## 🤗 Running on Hugging Face Spaces

VitalEye is designed to run seamlessly on **Hugging Face Spaces** (Free CPU Tier).

### 1. Configuration
The `Dockerfile` is pre-configured to:
- Install `ffmpeg` and system libraries.
- Set up a non-root user (`user:1000`).
- Optimize execution using **OpenVINO** for CPU-only environments.

### 2. Deployment Steps
1. Create a new **Docker** Space on Hugging Face.
2. Upload all files from this directory to the Space.
3. Ensure the `README.md` (metadata) includes `sdk: docker`.
4. The Space will automatically build and launch the Gradio app.

---

## 🛠 Technical Specifications
| Feature | Specification |
| :--- | :--- |
| **Detection** | YOLOv8 (Medium) |
| **Extraction** | PhysFormerX (3D-CNN + SE Attention) |
| **Input Shape** | 300 Frames x 64x64 x 6 Channels |
| **Sliding Window** | 10s Window / 1s Stride |
| **HR Range** | 45 - 150 BPM |
| **RR Range** | 6 - 30 BRPM |
| **CPU Speedup** | OpenVINO Enabled |
