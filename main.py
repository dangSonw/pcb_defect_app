#!/usr/bin/python3
import gradio as gr
from ui import tab_config, tab_train, tab_dataset, tab_export, tab_run
from core.dataset_manager import load_system_config
from core.camera_service import get_camera

cfg = load_system_config()

camera_available = False
try:
    camera = get_camera()
    if camera.start():
        camera_available = True
        print("[Main] ✓ Camera CSI đã khởi động thành công")
    else:
        print("[Main] ⚠ Cảnh báo: Camera CSI không khả dụng")
        print("[Main] → Sử dụng chế độ Upload ảnh từ máy tính")
except Exception as e:
    print(f"[Main] ⚠ Cảnh báo: Lỗi khởi động camera: {e}")
    print("[Main] → Sử dụng chế độ Upload ảnh từ máy tính")

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
        with gr.TabItem("1. Cài đặt Hệ thống"):
            tab_config.render(sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, cfg)
        with gr.TabItem("2. Huấn luyện (Train)"):
            tab_train.render(sys_device, sys_model_type, sys_model_path, sys_dataset_path)
        with gr.TabItem("3. Quản lý Dataset"): 
            # Tab 3 cập nhật thêm tham số camera_available
            tab_dataset.render(sys_dataset_path, sys_device, camera_available)
        with gr.TabItem("4. Xuất Model (Export)"):
            tab_export.render(sys_device, sys_model_path)
        with gr.TabItem("5. Rà quét PCB (Inference)"):
            tab_run.render(sys_device, sys_model_path, sys_output_dir, camera_available, app)

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
            print("[Main] ✓ Camera CSI đã tắt")
        except Exception as e:
            print(f"[Main] ⚠ Lỗi tắt camera: {e}")