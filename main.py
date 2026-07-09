#!/usr/bin/python3
import gradio as gr
from ui import tab_config, tab_train, tab_dataset, tab_export, tab_run, tab_pipeline
from core.dataset_manager import load_system_config
from core.camera_service import get_camera

cfg = load_system_config()

camera_available = False
try:
    camera = get_camera()
    if camera.start():
        camera_available = True
        print("[Main] ✓ CSI Camera started successfully")
    else:
        print("[Main] ⚠ Warning: CSI Camera is not available")
        print("[Main] → Using Image Upload mode from computer")
except Exception as e:
    print(f"[Main] ⚠ Warning: Camera startup error: {e}")
    print("[Main] → Using Image Upload mode from computer")

siemens_theme = gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate"
    )

with gr.Blocks(title="PCB Defect QA/QC") as app:
    gr.Markdown("(QA/QC)")
    
    sys_device = gr.State(cfg["device"])
    sys_model_type = gr.State(cfg["model_type"])
    sys_model_path = gr.State(cfg.get("model_path", ""))
    sys_dataset_path = gr.State(cfg["dataset_path"])
    sys_output_dir = gr.State(cfg["output_dir"])
    
    with gr.Tabs():
        with gr.TabItem("1. System Settings"):
            tab_config.render(sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, cfg)
        with gr.TabItem("2. Training"):
            tab_train.render(sys_device, sys_model_type, sys_model_path, sys_dataset_path)
        with gr.TabItem("3. Dataset Management"): 
            # Tab 3 cập nhật thêm tham số camera_available
            tab_dataset.render(sys_dataset_path, sys_device, camera_available)
        with gr.TabItem("4. Model Export"):
            tab_export.render(sys_device, sys_model_path)
        with gr.TabItem("5. PCB Detection"):
            tab_run.render(sys_device, sys_model_path, sys_output_dir, camera_available, app)
        with gr.TabItem("6. AI Pipeline"):
            tab_pipeline.render()

if __name__ == "__main__":
    try:
        app.launch(
            server_name="0.0.0.0", 
            theme=siemens_theme
        )
    finally:
        # Dừng camera khi ứng dụng thoát
        try:
            camera = get_camera()
            camera.stop()
            print("[Main] ✓ CSI Camera stopped")
        except Exception as e:
            print(f"[Main] ⚠ Error stopping camera: {e}")