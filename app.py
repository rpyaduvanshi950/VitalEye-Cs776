import gradio as gr
import os
import sliding_pipeline
from datetime import datetime
from types import SimpleNamespace

# Model parameters config
yolo_weights = "models/yolov8/weights/best.pt"
physformer_weights = "models/physformer.pt"

def process_video(video_path):
    if video_path is None:
        raise gr.Error("Please upload a video file.")
        
    # Generate unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    out_mp4 = os.path.join(output_dir, "annotated_output.mp4")
    
    # Construct programmatic argument bindings
    args = SimpleNamespace(
        video=video_path,
        yolo_weights=yolo_weights,
        physformer_weights=physformer_weights,
        hf_repo="pushpender-23/yolo-cs776-model",
        yolo_hf_file="train_medium_v1/weights/best.pt",
        physformer_hf_file="physformer.pt",
        output=out_mp4,
        window_sec=10,
        stride_sec=1
    )
    
    # Execute native analytics pipeline securely
    try:
        gr.Info("Initializing YOLO face tracking & native 3D-CNN continuous sequence...")
        final_bpm, final_rr, plot_path, annotated_video = sliding_pipeline.main(args)
    except Exception as e:
        raise gr.Error(f"Pipeline Execution Failed: {str(e)}")
        
    bpm_text = f"{final_bpm:.1f} BPM"
    rr_text = f"{final_rr:.1f} BRPM"
    gr.Info("Processing successfully completed!")
    
    return bpm_text, rr_text, plot_path, annotated_video

# ====================================
# Gradio UI Blocks definition
# ====================================
with gr.Blocks(title="PhysformerX rPPG Analytics") as demo:
    gr.Markdown("# 🫀 PhysformerX rPPG & YOLOv8 Analytics")
    gr.Markdown("Upload a video to perform causal body-part tracking and compute Heart Rate & Respiratory Rate continuously.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.File(label="Input Video (Accepts AVI, MP4, MKV, MOV...)", file_types=["video", ".avi", ".mkv", ".mov", ".mp4"], type="filepath")
            submit_btn = gr.Button("Calculate HR & RR Analytics", variant="primary")
        with gr.Column():
            bpm_output = gr.Textbox(label="🌟 Global Estimated Heart Rate (BPM)")
            rr_output = gr.Textbox(label="🌟 Global Estimated Respiratory Rate (BRPM)")
            
    with gr.Row():
        annotated_output = gr.Video(label="Processed Causal Tracking Video")
        waveform_output = gr.Image(label="Continuous rPPG Waveform Signal")
        
    submit_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[bpm_output, rr_output, waveform_output, annotated_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
