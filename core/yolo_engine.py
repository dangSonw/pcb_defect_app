import os

def train_yolo_model(model_type, model_path, data_yaml_path, epochs, batch_size, imgsz, device):
    try:
        if not data_yaml_path or not os.path.exists(data_yaml_path):
            return "LOI: Khong tim thay file data.yaml. Vui long kiem tra tab Train/Dataset."
        if int(epochs) <= 0 or int(batch_size) <= 0 or int(imgsz) <= 0:
            return "LOI: Tham so train khong hop le (epochs/batch/imgsz phai > 0)."
        
        train_device = "cpu"
        if "GPU" in device or "cuda" in device.lower():
            train_device = "0" 

        # Lazy load: Chỉ import thư viện YOLO khi người dùng thực sự bấm nút Train
        from ultralytics import YOLO
        if model_path:
            if not os.path.exists(model_path):
                return f"LOI: Khong tim thay model tai {model_path}"
            model_name = model_path
        else:
            model_name = f"{model_type}.pt"
        model = YOLO(model_name)

        results = model.train(
            data=data_yaml_path,
            epochs=int(epochs),
            batch=int(batch_size),
            imgsz=int(imgsz),
            device=train_device,
            project="outputs/train",
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
        ]
        return "\n".join(log_lines)

    except Exception as e:
        return f"LOI TRAIN:\n{str(e)}"