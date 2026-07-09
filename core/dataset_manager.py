import os, cv2, time, shutil, random, json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")


def _is_remote_path(path):
    if not path:
        return False
    path = str(path).strip()
    if not path:
        return False
    lower_path = path.lower()
    return (
        path.startswith("\\\\")
        or path.startswith("//")
        or lower_path.startswith("smb://")
        or lower_path.startswith("smb:\\\\")
    )


def _normalize_remote_path(path):
    if not path:
        return ""
    path = str(path).strip()
    if not path:
        return ""
    if path.lower().startswith("smb://"):
        return "//" + path[len("smb://"):].lstrip("/")
    if path.lower().startswith("smb:\\\\"):
        return "//" + path[len("smb:\\\\"):].replace("\\", "/").lstrip("/")
    if path.startswith("\\\\") or path.startswith("//"):
        return path.replace("\\", "/")
    return path


def resolve_project_path(input_path):
    """Chuẩn hoá đường dẫn tương đối/absolute trong dự án, đồng thời giữ nguyên đường dẫn SMB/UNC."""
    if not input_path:
        return ""
    path = str(input_path).strip()
    if not path:
        return ""

    if _is_remote_path(path):
        return _normalize_remote_path(path)

    # Nếu đã là đường dẫn tuyệt đối, giữ nguyên khi tồn tại.
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        basename = os.path.basename(path)
        candidate = os.path.join(DEFAULT_WEIGHTS_DIR, basename)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, PROJECT_ROOT)
        candidate = os.path.join(PROJECT_ROOT, basename)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, PROJECT_ROOT)
        return path

    # Nếu là đường dẫn tương đối, ưu tiên nội tại dự án.
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return os.path.normpath(path)
    if os.path.exists(path):
        return path
    basename = os.path.basename(path)
    candidate = os.path.join(DEFAULT_WEIGHTS_DIR, basename)
    if os.path.exists(candidate):
        return os.path.relpath(candidate, PROJECT_ROOT)
    return path


def load_system_config():
    """Nạp cấu hình hệ thống từ file json."""
    default_config = {
        "device": "cpu",
        "model": "yolov11s-obb",
        "dataset": "./dataset",
        "raw_data_dir": "dataset/raw_data",
        "output": "outputs",
        "log_data_base_dir": "outputs/LogData",
        "log_image_base_dir": "outputs/LogImage",
        "train_output_dir": "outputs/train",
        "export_output_dir": "weights",
        "source_model_pt": "weights/best.pt",
        "sahi_slice_size": 640,
        "sahi_overlap_ratio": 0.2,
        "defect_classes": "Empty, Excess Solder, Exposed Copper, Misaligned Header, Missing Component, Scratched, Solder Bridge",
        "cam_width": 3280,
        "cam_height": 2464,
        "cam_quality": 95,
        "defect_styles": {},
        "result_status_style": {
            "ok_color_bgr": [0, 255, 0],
            "ng_color_bgr": [0, 0, 255],
            "font_scale": 4,
            "thickness": 10,
            "position": [50, 120]
        }
    }
    config_file = "system_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
        except Exception:
            pass
    # Backward compatibility with newer config keys used by UI.
    if "model_type" not in default_config and "model" in default_config:
        default_config["model_type"] = default_config["model"]
    if "dataset_path" not in default_config and "dataset" in default_config:
        default_config["dataset_path"] = default_config["dataset"]
    if "output_dir" not in default_config and "output" in default_config:
        default_config["output_dir"] = default_config["output"]
    if "model_path" not in default_config:
        default_config["model_path"] = ""
    if "log_data_base_dir" not in default_config:
        default_config["log_data_base_dir"] = "outputs/LogData"
    if "log_image_base_dir" not in default_config:
        default_config["log_image_base_dir"] = "outputs/LogImage"
    if "train_output_dir" not in default_config:
        default_config["train_output_dir"] = "outputs/train"
    if "export_output_dir" not in default_config:
        default_config["export_output_dir"] = "weights"
    if "source_model_pt" not in default_config:
        default_config["source_model_pt"] = "weights/best.pt"
    if "raw_data_dir" not in default_config:
        default_config["raw_data_dir"] = "dataset/raw_data"
    if "defect_styles" not in default_config:
        default_config["defect_styles"] = {}
    if "result_status_style" not in default_config:
        default_config["result_status_style"] = {
            "ok_color_bgr": [0, 255, 0],
            "ng_color_bgr": [0, 0, 255],
            "font_scale": 4,
            "thickness": 10,
            "position": [50, 120]
        }
    default_config["model_path"] = resolve_project_path(default_config["model_path"])
    default_config["dataset_path"] = resolve_project_path(default_config["dataset_path"])
    default_config["output_dir"] = resolve_project_path(default_config["output_dir"])
    default_config["log_data_base_dir"] = resolve_project_path(default_config["log_data_base_dir"])
    default_config["log_image_base_dir"] = resolve_project_path(default_config["log_image_base_dir"])
    default_config["train_output_dir"] = resolve_project_path(default_config["train_output_dir"])
    default_config["export_output_dir"] = resolve_project_path(default_config["export_output_dir"])
    default_config["raw_data_dir"] = resolve_project_path(default_config["raw_data_dir"])
    default_config["source_model_pt"] = resolve_project_path(default_config.get("source_model_pt", ""))
    return default_config


def save_system_config(device, model_type, model_path, dataset_path, output_dir, sahi_slice_size=640, sahi_overlap_ratio=0.2, defect_classes="", cam_width=3280, cam_height=2464, cam_quality=95):
    """Lưu cấu hình hệ thống theo schema mới và giữ key cũ để tương thích."""
    output_dir = str(output_dir or "outputs").strip()
    normalized_output_dir = resolve_project_path(output_dir) or output_dir
    normalized_dataset_path = resolve_project_path(str(dataset_path or "./dataset").strip()) or str(dataset_path or "./dataset").strip()
    payload = {
        "device": str(device or "CPU").strip(),
        "model_type": str(model_type or "yolov11s-obb").strip(),
        "model_path": str(model_path or "").strip(),
        "dataset_path": normalized_dataset_path,
        "output_dir": normalized_output_dir,
        "log_data_base_dir": os.path.join(normalized_output_dir, "LogData"),
        "log_image_base_dir": os.path.join(normalized_output_dir, "LogImage"),
        "train_output_dir": os.path.join(normalized_output_dir, "train"),
        "export_output_dir": "weights",
        "raw_data_dir": os.path.join(normalized_dataset_path, "raw_data"),
        "sahi_slice_size": int(sahi_slice_size) if sahi_slice_size else 640,
        "sahi_overlap_ratio": float(sahi_overlap_ratio) if sahi_overlap_ratio else 0.2,
        "defect_classes": str(defect_classes).strip(),
        "cam_width": int(cam_width) if cam_width else 3280,
        "cam_height": int(cam_height) if cam_height else 2464,
        "cam_quality": int(cam_quality) if cam_quality else 95
    }
    # Keep old keys so older code can still read.
    payload["model"] = payload["model_type"]
    payload["dataset"] = payload["dataset_path"]
    payload["output"] = payload["output_dir"]
    # Preserve status style config if already present in existing system_config.json
    try:
        if os.path.exists("system_config.json"):
            with open("system_config.json", "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
                if "result_status_style" in existing_cfg:
                    payload["result_status_style"] = existing_cfg["result_status_style"]
    except Exception:
        pass
    if "result_status_style" not in payload:
        payload["result_status_style"] = {
            "ok_color_bgr": [0, 255, 0],
            "ng_color_bgr": [0, 0, 255],
            "font_scale": 4,
            "thickness": 10,
            "position": [50, 120]
        }
    # Ensure we store a default source .pt model in config
    if payload.get("model_path") and payload.get("model_path").endswith('.pt'):
        payload["source_model_pt"] = payload["model_path"]
    else:
        payload["source_model_pt"] = os.path.join("weights", "best.pt")
    with open("system_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

def save_annotation(image, annotation_list, dataset_path, used_classes):
    """Lưu ảnh và nhãn YOLO, tự tăng số thứ tự nếu trùng tên để bảo vệ dữ liệu."""
    if not dataset_path or image is None:
        return "LỖI: Chưa có dữ liệu ảnh.", used_classes
    if annotation_list is None:
        annotation_list = []
    try:
        # Lấy raw_data_dir từ config
        cfg = load_system_config()
        raw_dir = cfg.get("raw_data_dir", "dataset/raw_data")
        raw_dir = resolve_project_path(raw_dir)
        
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

def split_dataset(dataset_path, train_r, val_r, test_r, bg_ratio=10):
    """Phân chia dữ liệu train/val/test, có hỗ trợ lọc ảnh background theo tỉ lệ."""
    if (train_r + val_r + test_r) != 100: return "LỖI: Tổng tỉ lệ phải bằng 100%."
    
    # Lấy raw_data_dir từ config
    cfg = load_system_config()
    raw_dir = cfg.get("raw_data_dir", "dataset/raw_data")
    raw_dir = resolve_project_path(raw_dir)
    
    if not os.path.exists(raw_dir): return "LỖI: Không có raw_data."
    try:
        images = [f for f in os.listdir(os.path.join(raw_dir, "images")) if f.endswith(('.jpg', '.png'))]
        if not images: return "LỖI: Thư mục ảnh rỗng."
        
        obj_images = []
        bg_images = []
        for f in images:
            lbl_file = f.rsplit('.', 1)[0] + ".txt"
            lbl_path = os.path.join(raw_dir, "labels", lbl_file)
            has_obj = False
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as lf:
                    if lf.read().strip():
                        has_obj = True
            if has_obj:
                obj_images.append(f)
            else:
                bg_images.append(f)
                
        keep_bg_count = int(len(obj_images) * (bg_ratio / 100.0))
        if keep_bg_count == 0 and bg_ratio > 0 and len(bg_images) > 0:
            keep_bg_count = 1
            
        random.shuffle(bg_images)
        kept_bg = bg_images[:keep_bg_count]
        
        filtered_images = obj_images + kept_bg
        random.shuffle(filtered_images)
        
        total = len(filtered_images)
        t_idx, v_idx = int(total * train_r / 100), int(total * (train_r + val_r) / 100)
        splits = {"train": filtered_images[:t_idx], "val": filtered_images[t_idx:v_idx], "test": filtered_images[v_idx:]}
        
        for s_name, files in splits.items():
            os.makedirs(os.path.join(dataset_path, s_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, s_name, "labels"), exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(raw_dir, "images", f), os.path.join(dataset_path, s_name, "images", f))
                lbl = f.rsplit('.', 1)[0] + ".txt"
                shutil.copy(os.path.join(raw_dir, "labels", lbl), os.path.join(dataset_path, s_name, "labels", lbl))
        
        # Tự động tạo file data.yaml cho YOLO
        classes_file = os.path.join(raw_dir, "classes.txt")
        class_names = []
        if os.path.exists(classes_file):
            with open(classes_file, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]
                
        yaml_content = (
            f"path: {os.path.abspath(dataset_path)}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        with open(os.path.join(dataset_path, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content)

        return f"Đã chia {total} mẫu thành công (Gồm {len(obj_images)} ảnh chứa lỗi và {len(kept_bg)} ảnh background)."
    except Exception as e: return f"LỖI CHIA DỮ LIỆU: {str(e)}"

def auto_slice_image(image_path, enable_sahi=False):
    """Cắt ảnh tự động theo cấu hình SAHI nếu được kích hoạt."""
    if not enable_sahi: return [image_path]
    
    cfg = load_system_config()
    slice_size = int(cfg.get("sahi_slice_size", 640))
    overlap_ratio = float(cfg.get("sahi_overlap_ratio", 0.2))
    
    img = cv2.imread(image_path)
    if img is None: return [image_path]
    h, w = img.shape[:2]
    if h <= slice_size and w <= slice_size: return [image_path]
    
    slices = []
    stride = int(slice_size * (1 - overlap_ratio))
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path)
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            y1, x1 = y, x
            y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
            patch = img[y1:y2, x1:x2]
            patch_path = os.path.join(out_dir, f"{base_name}_sahi_{y1}_{x1}.jpg")
            cv2.imwrite(patch_path, patch)
            slices.append(patch_path)
            if x2 == w: break
            x += stride
        if y2 == h: break
        y += stride
    return slices

def save_raw_image_to_log(image, status_folder="OK"):
    """
    Lưu nhanh ảnh chụp vào thư mục log (được cấu hình trong system_config.json), 
    bỏ qua quy trình đánh nhãn và phân chia dataset.
    """
    if image is None:
        return "LỖI: Chưa có dữ liệu ảnh."

    try:
        # Lấy log image directory từ config
        cfg = load_system_config()
        log_image_base_dir = cfg.get("log_image_base_dir", "outputs/LogImage")
        log_image_base_dir = resolve_project_path(log_image_base_dir)
        
        # Xây dựng đường dẫn: log_image_base_dir/<status_folder>
        log_img_dir = os.path.join(log_image_base_dir, str(status_folder))
        
        # Tạo thư mục nếu chưa tồn tại (exist_ok=True giúp không báo lỗi nếu đã có)
        os.makedirs(log_img_dir, exist_ok=True)

        # Tạo tên file duy nhất bằng timestamp (nhân 1000 để lấy tới mili-giây, chống trùng tuyệt đối khi chụp liên tục)
        filename = f"log_{status_folder}_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(log_img_dir, filename)

        # Chuyển đổi RGB sang BGR để OpenCV lưu đúng màu (đồng bộ với logic của save_annotation hiện tại)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ghi file
        success = cv2.imwrite(filepath, bgr_image)

        if success:
            return f"THÀNH CÔNG: Đã lưu ảnh vào {filepath}"
        else:
            return f"LỖI: Không thể ghi file. Hãy kiểm tra lại quyền ghi (Write Permission) của thư mục."

    except Exception as e:
        return f"LỖI LƯU ẢNH LOG: {str(e)}"