import os, cv2, time, shutil, random, json

def load_system_config():
    """Nạp cấu hình hệ thống từ file json."""
    default_config = {
        "device": "cpu",
        "model": "yolov11s-obb",
        "dataset": "./dataset",
        "output": "./outputs"
    }
    config_file = "system_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
        except Exception: pass
    # Backward compatibility with newer config keys used by UI.
    if "model_type" not in default_config and "model" in default_config:
        default_config["model_type"] = default_config["model"]
    if "dataset_path" not in default_config and "dataset" in default_config:
        default_config["dataset_path"] = default_config["dataset"]
    if "output_dir" not in default_config and "output" in default_config:
        default_config["output_dir"] = default_config["output"]
    if "model_path" not in default_config:
        default_config["model_path"] = ""
    return default_config


def save_system_config(device, model_type, model_path, dataset_path, output_dir):
    """Lưu cấu hình hệ thống theo schema mới và giữ key cũ để tương thích."""
    payload = {
        "device": str(device or "CPU").strip(),
        "model_type": str(model_type or "yolov11s-obb").strip(),
        "model_path": str(model_path or "").strip(),
        "dataset_path": str(dataset_path or "./dataset").strip(),
        "output_dir": str(output_dir or "./outputs").strip(),
    }
    # Keep old keys so older code can still read.
    payload["model"] = payload["model_type"]
    payload["dataset"] = payload["dataset_path"]
    payload["output"] = payload["output_dir"]
    with open("system_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

def save_annotation(image, annotation_list, dataset_path, used_classes):
    """Lưu ảnh và nhãn YOLO, tự tăng số thứ tự nếu trùng tên để bảo vệ dữ liệu."""
    if not dataset_path or image is None or not annotation_list:
        return "LỖI: Chưa có dữ liệu gán nhãn.", used_classes
    try:
        raw_dir = os.path.join(dataset_path, "raw_data")
        img_dir = os.path.join(raw_dir, "images")
        lbl_dir = os.path.join(raw_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Logic kiểm tra và chống trùng tên file
        base_name = f"pcb_{int(time.time())}"
        filename = base_name
        counter = 1
        while os.path.exists(os.path.join(img_dir, f"{filename}.jpg")):
            filename = f"{base_name}_{counter}"
            counter += 1

        # Lưu file ảnh
        cv2.imwrite(os.path.join(img_dir, f"{filename}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Đồng bộ danh sách Class từ classes.txt
        classes_file = os.path.join(raw_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, "r", encoding="utf-8") as f:
                for line in f:
                    c = line.strip()
                    if c and c not in used_classes: used_classes.append(c)

        h, w = image.shape[:2]
        yolo_lines = []
        for ann in annotation_list:
            b = ann['box']
            # Tọa độ YOLO chuẩn hóa
            x_c = (min(b[0], b[2]) + abs(b[0] - b[2]) / 2) / w
            y_c = (min(b[1], b[3]) + abs(b[1] - b[3]) / 2) / h
            width = abs(b[0] - b[2]) / w
            height = abs(b[1] - b[3]) / h
            
            if ann['label'] not in used_classes: 
                used_classes.append(ann['label'])
            class_id = used_classes.index(ann['label'])
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}")

        # Ghi file classes và nhãn
        with open(classes_file, "w", encoding="utf-8") as f:
            f.write("\n".join(used_classes))
        with open(os.path.join(lbl_dir, f"{filename}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))
            
        return f"THÀNH CÔNG: Đã lưu {filename}.jpg", used_classes
    except Exception as e: 
        return f"LỖI: {str(e)}", used_classes

def split_dataset(dataset_path, train_r, val_r, test_r):
    """Phân chia dữ liệu train/val/test."""
    if (train_r + val_r + test_r) != 100: return "LỖI: Tổng tỉ lệ phải bằng 100%."
    raw_dir = os.path.join(dataset_path, "raw_data")
    if not os.path.exists(raw_dir): return "LỖI: Không có raw_data."
    try:
        images = [f for f in os.listdir(os.path.join(raw_dir, "images")) if f.endswith(('.jpg', '.png'))]
        if not images: return "LỖI: Thư mục ảnh rỗng."
        random.shuffle(images)
        total = len(images)
        t_idx, v_idx = int(total * train_r / 100), int(total * (train_r + val_r) / 100)
        splits = {"train": images[:t_idx], "val": images[t_idx:v_idx], "test": images[v_idx:]}
        
        for s_name, files in splits.items():
            os.makedirs(os.path.join(dataset_path, s_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, s_name, "labels"), exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(raw_dir, "images", f), os.path.join(dataset_path, s_name, "images", f))
                lbl = f.rsplit('.', 1)[0] + ".txt"
                shutil.copy(os.path.join(raw_dir, "labels", lbl), os.path.join(dataset_path, s_name, "labels", lbl))
        return f"Đã chia {total} mẫu thành công."
    except Exception as e: return f"LỖI CHIA DỮ LIỆU: {str(e)}"