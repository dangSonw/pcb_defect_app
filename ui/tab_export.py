import gradio as gr
import tkinter as tk
from tkinter import filedialog
from core.exporter import export_model

def open_file_dialog():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Chọn file Model gốc (.pt)", filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
    root.destroy()
    return file_path if file_path else ""

def start_exporting(sys_device, model_path_input, sys_model_path, export_format, use_half, use_int8, imgsz):
    target_path = model_path_input if model_path_input else sys_model_path
    if not target_path:
        return sys_model_path, "LOI: Chua co duong dan model. Hay chon file hoac cau hinh o Tab 1."
    log_text, new_model_path = export_model(target_path, export_format, use_half, use_int8, imgsz, sys_device)
    return new_model_path, log_text

def render(sys_device, sys_model_path):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Thông số Biên dịch")
            ui_export_format = gr.Dropdown(choices=["TensorRT (.engine)", "ONNX (.onnx)"], value="TensorRT (.engine)", label="Định dạng đầu ra")
            ui_use_half = gr.Checkbox(value=True, label="FP16 Half-Precision")
            ui_use_int8 = gr.Checkbox(value=False, label="INT8 Precision")
            ui_imgsz = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Kích thước ảnh")

        with gr.Column(scale=2):
            gr.Markdown("### Nguồn dữ liệu")
            with gr.Row():
                ui_target_model = gr.Textbox(placeholder="Đường dẫn sẽ lấy từ Cài đặt nếu để trống...", scale=5, label="File Model gốc (.pt)")
                btn_browse_target = gr.Button("Duyệt File", scale=1)
                
            export_btn = gr.Button("BẮT ĐẦU BIÊN DỊCH", variant="primary")
            ui_log_output = gr.Textbox(label="Nhật ký", lines=8, interactive=False)

    btn_browse_target.click(fn=open_file_dialog, inputs=[], outputs=[ui_target_model])
    export_btn.click(
        fn=start_exporting,
        inputs=[sys_device, ui_target_model, sys_model_path, ui_export_format, ui_use_half, ui_use_int8, ui_imgsz],
        outputs=[sys_model_path, ui_log_output]
    )