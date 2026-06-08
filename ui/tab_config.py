import gradio as gr
import tkinter as tk
from tkinter import filedialog
from core.dataset_manager import save_system_config

# --- CÁC HÀM TIỆN ÍCH ---

def auto_detect_hardware():
    """Tự động kiểm tra phần cứng hiện có"""
    try:
        import torch
        import subprocess
        
        # Kiểm tra Jetson
        is_jetson = False
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'R36' in content or 'Jetson' in content.lower():
                    is_jetson = True
        except:
            pass
        
        # Kiểm tra CUDA
        cuda_available = torch.cuda.is_available()
        
        # Kiểm tra TensorRT
        tensorrt_available = False
        try:
            import tensorrt
            tensorrt_available = True
        except:
            pass
        
        if is_jetson:
            if cuda_available and tensorrt_available:
                return "Jetson GPU (TensorRT/CUDA)"
            elif cuda_available:
                return "Jetson GPU (CUDA)"
            else:
                return "Jetson (CPU mode - CUDA driver mismatch)"
        elif cuda_available:
            device_name = torch.cuda.get_device_name(0)
            return f"GPU: {device_name}"
        else:
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
    folder_path = filedialog.askdirectory(title="Chọn Thư mục")
    root.destroy()
    return folder_path if folder_path else ""

def get_hardware_info():
    """Lấy thông tin chi tiết về phần cứng"""
    try:
        import torch
        import subprocess
        
        info_lines = ["=== HARDWARE INFO ==="]
        
        # PyTorch info
        info_lines.append(f"PyTorch: {torch.__version__}")
        info_lines.append(f"CUDA available: {torch.cuda.is_available()}")
        info_lines.append(f"CUDA version: {torch.version.cuda}")
        info_lines.append(f"CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                info_lines.append(f"GPU {i}: {device_name} ({memory_gb:.1f} GB)")
        
        # Jetson info
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read().strip()
                info_lines.append(f"Jetson release: {content}")
        except:
            info_lines.append("Not a Jetson device")
        
        # TensorRT info
        try:
            import tensorrt
            info_lines.append(f"TensorRT: {tensorrt.__version__}")
        except:
            info_lines.append("TensorRT: Not available")
        
        # NVIDIA SMI
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info_lines.append("NVIDIA SMI:")
                for line in result.stdout.strip().split('\n'):
                    info_lines.append(f"  {line}")
        except:
            info_lines.append("NVIDIA SMI: Not available")
            
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error getting hardware info: {e}"

def save_settings(device, model_type, model_path, dataset_path, output_dir, sahi_slice_size, sahi_overlap_ratio, defect_classes, cam_width, cam_height, cam_quality):
    try:
        payload = save_system_config(device, model_type, model_path, dataset_path, output_dir, sahi_slice_size, sahi_overlap_ratio, defect_classes, cam_width, cam_height, cam_quality)
        msg = f"Da luu cau hinh: {payload['device']} | {payload['model_type']}"
        return (
            payload["device"],
            payload["model_type"],
            payload["model_path"],
            payload["dataset_path"],
            payload["output_dir"],
            payload["sahi_slice_size"],
            payload["sahi_overlap_ratio"],
            payload["defect_classes"],
            payload["cam_width"],
            payload["cam_height"],
            payload["cam_quality"],
            msg,
        )
    except Exception as e:
        msg = f"Loi luu cau hinh: {e}"
        return device, model_type, model_path, dataset_path, output_dir, sahi_slice_size, sahi_overlap_ratio, defect_classes, cam_width, cam_height, cam_quality, msg

# --- HÀM VẼ GIAO DIỆN ---

def render(sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, cfg):
    detected_hw = auto_detect_hardware()
    
    # Tạo danh sách device choices dựa trên hardware detected
    device_choices = ["CPU"]
    if "Jetson" in detected_hw:
        device_choices.extend(["Jetson GPU (CUDA)", "Jetson GPU (TensorRT/CUDA)"])
    elif "GPU" in detected_hw:
        device_choices.append(detected_hw)
        device_choices.append("cuda:0")
    else:
        device_choices.append("cuda:0")  # Fallback
    
    model_choices = [
        "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x",
        "yolov11n-obb", "yolov11s-obb", "yolov11m-obb", "yolov11l-obb", "yolov11x-obb",
        "yolov26n", "yolov26s", "yolov26m", "yolov26l"
    ]
    
    with gr.Row():
        # CỘT 1: THIẾT LẬP HỆ THỐNG
        with gr.Column(scale=1):
            gr.Markdown("### Cấu hình AI & Phần cứng")
            
            gr.Markdown(f"**Phát hiện phần cứng:** {detected_hw}")
            
            ui_device = gr.Dropdown(
                choices=device_choices, 
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
                ui_output_dir = gr.Textbox(scale=5, value=cfg.get("output_dir", "outputs"), label="Thư mục lưu báo cáo")
                btn_browse_out = gr.Button("Duyệt Thư mục", scale=1)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Cấu hình Thuật toán SAHI")
            ui_sahi_slice_size = gr.Number(value=cfg.get("sahi_slice_size", 640), label="Kích thước cắt ảnh (Slice Size)", precision=0)
            ui_sahi_overlap_ratio = gr.Slider(minimum=0.0, maximum=0.9, value=cfg.get("sahi_overlap_ratio", 0.2), step=0.05, label="Tỉ lệ chồng lấn (Overlap Ratio)")
        with gr.Column(scale=2):
            gr.Markdown("### Cấu hình Báo cáo Lỗi đầu ra")
            ui_defect_classes = gr.Textbox(value=cfg.get("defect_classes", "short_circuit, open_circuit, missing_hole, mouse_bite, spur, copper_salvage"), label="Danh sách Class Lỗi (cách nhau bằng dấu phẩy)")

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Cấu hình Camera CSI (Kích thước & Dung lượng)")
            with gr.Row():
                ui_cam_width = gr.Number(value=cfg.get("cam_width", 3280), label="Chiều rộng (Width px)", precision=0)
                ui_cam_height = gr.Number(value=cfg.get("cam_height", 2464), label="Chiều cao (Height px)", precision=0)
            ui_cam_quality = gr.Slider(minimum=10, maximum=100, value=cfg.get("cam_quality", 95), step=5, label="Chất lượng JPEG (Quality %)")

    gr.Markdown("---")
    
    with gr.Row():
        check_hw_btn = gr.Button("🔍 Kiểm tra Hardware", variant="secondary", scale=1)
        save_btn = gr.Button("Lưu Cài Đặt Hệ Thống", variant="primary", scale=1)
        status_msg = gr.Textbox(label="Terminal Trạng thái", interactive=False, scale=2)

    # --- KẾT NỐI SỰ KIỆN ---
    check_hw_btn.click(fn=get_hardware_info, inputs=[], outputs=[status_msg])
    btn_browse_model.click(fn=open_file_dialog, inputs=[], outputs=[ui_model_path])
    btn_browse_data.click(fn=open_folder_dialog, inputs=[], outputs=[ui_dataset_path])
    btn_browse_out.click(fn=open_folder_dialog, inputs=[], outputs=[ui_output_dir])

    save_btn.click(
        fn=save_settings,
        inputs=[ui_device, ui_model_type, ui_model_path, ui_dataset_path, ui_output_dir, ui_sahi_slice_size, ui_sahi_overlap_ratio, ui_defect_classes, ui_cam_width, ui_cam_height, ui_cam_quality],
        outputs=[sys_device, sys_model_type, sys_model_path, sys_dataset_path, sys_output_dir, ui_sahi_slice_size, ui_sahi_overlap_ratio, ui_defect_classes, ui_cam_width, ui_cam_height, ui_cam_quality, status_msg]
    )