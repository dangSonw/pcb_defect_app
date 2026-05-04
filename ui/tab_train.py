import gradio as gr
import os
from core.yolo_engine import train_yolo_model

def start_training(sys_device, sys_model_type, sys_model_path, sys_dataset_path, data_yaml_path, epochs, batch_size, imgsz):
    yaml_path = data_yaml_path.strip() if data_yaml_path else os.path.join(sys_dataset_path or "", "data.yaml")
    result_log = train_yolo_model(
        model_type=sys_model_type,
        model_path=sys_model_path,
        data_yaml_path=yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        device=sys_device,
    )
    return result_log

def render(sys_device, sys_model_type, sys_model_path, sys_dataset_path):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Thông số Huấn luyện")
            ui_epochs = gr.Slider(minimum=1, maximum=1000, value=100, step=1, label="Số vòng lặp (Epochs)")
            ui_batch_size = gr.Slider(minimum=1, maximum=128, value=16, step=1, label="Kích thước lô (Batch Size)")
            ui_imgsz = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Kích thước ảnh đầu vào (Image Size)")
            train_btn = gr.Button("BẮT ĐẦU HUẤN LUYỆN", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Giám sát Hệ thống")
            ui_data_yaml = gr.Textbox(label="Duong dan data.yaml", value="dataset/data.yaml")
            ui_log_output = gr.Textbox(label="Nhật ký", lines=12, interactive=False, placeholder="Tiến trình huấn luyện sẽ hiển thị ở đây...")

    train_btn.click(
        fn=start_training,
        inputs=[sys_device, sys_model_type, sys_model_path, sys_dataset_path, ui_data_yaml, ui_epochs, ui_batch_size, ui_imgsz],
        outputs=[ui_log_output]
    )