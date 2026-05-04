import gradio as gr
from ui import tab_config, tab_train, tab_dataset, tab_export, tab_run
from core.dataset_manager import load_system_config

# Nạp cấu hình cũ khi khởi động 
cfg = load_system_config()

with gr.Blocks(title="PCB Defect QA/QC") as app:
    gr.Markdown("# Hệ thống Nhận diện Vi khuyết tật PCB (QA/QC)")
    
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
            # Tab 3 nhận 2 tham số để quản lý cấu hình và dữ liệu [cite: 242, 243]
            tab_dataset.render(sys_dataset_path, sys_device)
        with gr.TabItem("4. Xuất Model (Export)"):
            tab_export.render(sys_device, sys_model_path)
        with gr.TabItem("5. Rà quét PCB (Inference)"):
            tab_run.render(sys_device, sys_model_path, sys_output_dir)
        with gr.TabItem("6. Tìm Lỗi Chưa Biết (Anomaly)"):
            gr.Markdown("*(Tính năng Học tăng cường đang được phát triển...)*")
            # tab_anomaly.render(...)

if __name__ == "__main__":
    # Khôi phục màu xanh Siemens (Blue/Teal) [cite: 276]
    siemens_theme = gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate"
    )
    app.launch(
        server_name="127.0.0.1", 
        theme=siemens_theme
    )