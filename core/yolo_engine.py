import os
from core.dataset_manager import resolve_project_path, PROJECT_ROOT, load_system_config

def train_yolo_model(model_type, model_path, data_yaml_path, epochs, batch_size, imgsz, device):
    try:
        if not data_yaml_path or not os.path.exists(data_yaml_path):
            return "LOI: Khong tim thay file data.yaml. Vui long kiem tra tab Train/Dataset."
        if int(epochs) <= 0 or int(batch_size) <= 0 or int(imgsz) <= 0:
            return "LOI: Tham so train khong hop le (epochs/batch/imgsz phai > 0)."
        
        # Kiểm tra phần cứng chi tiết
        train_device = "cpu"
        hardware_info = []
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            hardware_info.append(f"PyTorch {torch.__version__}")
            
            # Kiểm tra Jetson
            is_jetson = False
            try:
                with open('/etc/nv_tegra_release', 'r') as f:
                    content = f.read()
                    if 'R36' in content or 'Jetson' in content.lower():
                        is_jetson = True
                        hardware_info.append("Jetson Nano detected")
            except:
                pass
            
            # Kiểm tra TensorRT
            tensorrt_available = False
            try:
                import tensorrt
                tensorrt_available = True
                hardware_info.append(f"TensorRT {tensorrt.__version__}")
            except:
                hardware_info.append("TensorRT not available")
            
            if cuda_available:
                hardware_info.append(f"CUDA {torch.version.cuda} available")
                device_name = torch.cuda.get_device_name(0)
                hardware_info.append(f"GPU: {device_name}")
            else:
                hardware_info.append("CUDA not available - using CPU")
                
        except Exception as e:
            hardware_info.append(f"Hardware check error: {e}")
            cuda_available = False
            tensorrt_available = False

        device_lower = str(device or "").lower()
        if cuda_available and ("gpu" in device_lower or "cuda" in device_lower or "jetson" in device_lower):
            train_device = "0"
            hardware_info.append("Training on GPU")
        else:
            train_device = "cpu"
            hardware_info.append("Training on CPU")

        # Lazy load: Chỉ import thư viện YOLO khi người dùng thực sự bấm nút Train
        from ultralytics import YOLO
        if model_path:
            model_name = resolve_project_path(model_path)
            if not os.path.exists(model_name):
                return f"LOI: Khong tim thay model tai {model_path}"
        else:
            fallback = os.path.join(PROJECT_ROOT, "weights", f"{model_type}.pt")
            if os.path.exists(fallback):
                model_name = fallback
            else:
                model_name = f"{model_type}.pt"
        
        hardware_info.append(f"Model: {model_name}")
        hardware_info.append(f"Device: {train_device}")
        
        model = YOLO(model_name)

        # Lấy train output directory từ config
        cfg = load_system_config()
        train_output_base = cfg.get("train_output_dir", "outputs/train")
        train_output_base = resolve_project_path(train_output_base)
        
        results = model.train(
            data=data_yaml_path,
            epochs=int(epochs),
            batch=int(batch_size),
            imgsz=int(imgsz),
            device=train_device,
            project=train_output_base,
            name=f"{model_type}_pcb",
            exist_ok=True
        )
        
        best_model_path = os.path.join(str(results.save_dir), "weights", "best.pt")
        log_lines = [
            "THANH CONG: Huan luyen hoan tat",
            f"Model source: {model_name}",
            f"Data yaml: {os.path.abspath(data_yaml_path)}",
            f"Device: {train_device}",
            f"Run dir: {results.save_dir}",
            f"Best model: {best_model_path}",
            "",
            "=== HARDWARE INFO ===",
        ] + hardware_info
        return "\n".join(log_lines)

    except Exception as e:
        return f"LOI TRAIN:\n{str(e)}"