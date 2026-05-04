import gradio as gr
import tkinter as tk
from tkinter import filedialog
from core.dataset_manager import save_system_config

# --- CÁC HÀM TIỆN ÍCH ---

def auto_detect_hardware():
    """Tự động kiểm tra phần cứng hiện có"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "Orin" in device_name or "Nano" in device_name or "Tegra" in device_name:
                return "Jetson GPU (TensorRT/CUDA)"
            return f"GPU: {device_name}"
        return "CPU"
    except ImportError:
        return "CPU (Chưa cài PyTorch)"

def open_file_dialog():
    """Mở cửa sổ chọn file cục bộ"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Chọn file Model", filetypes=[("Model files", "*.pt *.engine *.onnx"), ("All files", "*.*")])
    root.destroy()
    return file_path if file_path else ""

def open_folder_dialog():
    """Mở cửa sổ chọn thư mục cục bộ"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Chọn thư mục")
    root.destroy()
    return folder_path if folder_path else ""

def save_settings(device, model_type, model_path, dataset_path, output_dir):
    try:
        payload = save_system_config(device, model_type, model_path, dataset_path, output_dir)
        msg = f"Da luu cau hinh: {payload['device']} | {payload['model_type']}"
        return (
            payload["device"],
            payload["model_type"],
            payload["model_path"],
            payload["dataset_path"],
            payload["output_dir"],
            msg,
        )
    except Exception as e:
        msg = f"Loi luu cau hinh: {e}"
        return device, model_type, model_path, dataset_path, output_dir, msg

# --- HÀM VẼ GIAO DIỆN ---

def render(sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, cfg):
    detected_hw = auto_detect_hardware()
    
    model_choices = [
        "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x",
        "yolov11n-obb", "yolov11s-obb", "yolov11m-obb", "yolov11l-obb", "yolov11x-obb",
        "yolov26n", "yolov26s", "yolov26m", "yolov26l"
    ]
    
    with gr.Row():
        # CỘT 1: THIẾT LẬP HỆ THỐNG
        with gr.Column(scale=1):
            gr.Markdown("### Cấu hình AI & Phần cứng")
            
            ui_device = gr.Dropdown(
                choices=["CPU", detected_hw, "Jetson GPU (TensorRT)", "cuda:0"], 
                value=cfg.get("device", detected_hw), 
                label="Lựa chọn phần cứng", 
                allow_custom_value=True
            )
            
            ui_model_type = gr.Dropdown(
                choices=model_choices, 
                value=cfg.get("model_type", "yolov11s-obb"),
                label="Loại mô hình mặc định"
            )

        # CỘT 2: ĐƯỜNG DẪN DỮ LIỆU
        with gr.Column(scale=2):
            gr.Markdown("### Cấu hình Đường dẫn (I/O)")
            
            with gr.Row():
                ui_model_path = gr.Textbox(scale=5, label="Đường dẫn Model (.pt, .engine)", value=cfg.get("model_path", ""))
                btn_browse_model = gr.Button("Duyệt File", scale=1)
                
            with gr.Row():
                ui_dataset_path = gr.Textbox(scale=5, label="Đường dẫn Dataset (Thư mục)", value=cfg.get("dataset_path", "./dataset"))
                btn_browse_data = gr.Button("Duyệt Thư mục", scale=1)
                
            with gr.Row():
                ui_output_dir = gr.Textbox(scale=5, value=cfg.get("output_dir", "outputs/"), label="Thư mục lưu báo cáo")
                btn_browse_out = gr.Button("Duyệt Thư mục", scale=1)

    gr.Markdown("---")
    
    with gr.Row():
        save_btn = gr.Button("Lưu Cài Đặt Hệ Thống", variant="primary", scale=1)
        status_msg = gr.Textbox(label="Terminal Trạng thái", interactive=False, scale=2)

    # --- KẾT NỐI SỰ KIỆN ---
    btn_browse_model.click(fn=open_file_dialog, inputs=[], outputs=[ui_model_path])
    btn_browse_data.click(fn=open_folder_dialog, inputs=[], outputs=[ui_dataset_path])
    btn_browse_out.click(fn=open_folder_dialog, inputs=[], outputs=[ui_output_dir])

    save_btn.click(
        fn=save_settings,
        inputs=[ui_device, ui_model_type, ui_model_path, ui_dataset_path, ui_output_dir],
        outputs=[sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, status_msg]
    )