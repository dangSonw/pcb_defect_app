import gradio as gr
import tkinter as tk
from tkinter import filedialog
from core.exporter import export_model
from core.dataset_manager import load_system_config

def open_file_dialog():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select source Model file (.pt)", filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
    root.destroy()
    return file_path if file_path else ""

def start_exporting(sys_device, model_path_input, sys_model_path, export_format, use_half, use_int8, imgsz):
    # Prefer explicit user input, then sys_model_path, then config source_model_pt
    cfg = load_system_config()
    fallback = cfg.get("source_model_pt", "")
    target_path = model_path_input if model_path_input else (sys_model_path if sys_model_path else fallback)
    if not target_path:
        return sys_model_path, "ERROR: Model path not found. Please select a file or configure in Tab 1."
    log_text, new_model_path = export_model(target_path, export_format, use_half, use_int8, imgsz, sys_device)
    return new_model_path, log_text

def render(sys_device, sys_model_path):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Export Parameters")
            ui_export_format = gr.Dropdown(choices=["TensorRT (.engine)", "ONNX (.onnx)"], value="TensorRT (.engine)", label="Output Format")
            ui_use_half = gr.Checkbox(value=True, label="FP16 Half-Precision")
            ui_use_int8 = gr.Checkbox(value=False, label="INT8 Precision")
            ui_imgsz = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size")

        with gr.Column(scale=2):
            gr.Markdown("### Data Source")
            with gr.Row():
                cfg = load_system_config()
                ui_target_model = gr.Textbox(placeholder="Path will be taken from Settings if left empty...", scale=5, label="Source Model file (.pt)", value=cfg.get("source_model_pt", ""))
                btn_browse_target = gr.Button("Browse File", scale=1)
                
            export_btn = gr.Button("START EXPORT", variant="primary")
            ui_log_output = gr.Textbox(label="Log", lines=8, interactive=False)

    btn_browse_target.click(fn=open_file_dialog, inputs=[], outputs=[ui_target_model])
    export_btn.click(
        fn=start_exporting,
        inputs=[sys_device, ui_target_model, sys_model_path, ui_export_format, ui_use_half, ui_use_int8, ui_imgsz],
        outputs=[sys_model_path, ui_log_output]
    )