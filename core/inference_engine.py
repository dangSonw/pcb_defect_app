import os
import cv2
import time
import json
import datetime
import csv
import numpy as np
from ultralytics import YOLO

from core.dataset_manager import resolve_project_path
from core.dataset_manager import load_system_config
from utils.vision_slicer import infer_with_optional_slicing

# --- THÊM CACHE MODEL ĐỂ TỐI ƯU TỐC ĐỘ (TRÁNH LOAD LẠI MỖI LẦN QUÉT) ---
_cached_model_path = None
_cached_model = None

def _get_model(model_path):
    global _cached_model_path, _cached_model
    if _cached_model_path == model_path and _cached_model is not None:
        return _cached_model
        
    print(f"[AI] Đang nạp model vào RAM/VRAM: {model_path} ...")
    _cached_model = YOLO(model_path)
    _cached_model_path = model_path
    
    # Warm-up (Khởi động nóng) để TensorRT dịch tối ưu trong lần chạy đầu
    try:
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _cached_model.predict(dummy_img, verbose=False)
        print("[AI] Khởi động nóng (Warmup) thành công.")
    except Exception:
        pass
        
    return _cached_model

def _dbg(hypothesis_id, location, message, data):
    # Đã vô hiệu hóa để tránh việc liên tục mở/đóng file I/O gây chậm tốc độ xử lý
    pass


def _get_class_names(model):
    return model.names if isinstance(model.names, list) else [model.names[k] for k in sorted(model.names.keys())]


def _build_detections(raw_dets, class_names):
    detections = []
    for d in raw_dets:
        x1, y1, x2, y2, cls_id, score = d
        label = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(cls_id)
        detections.append(
            {
                "class_id": int(cls_id),
                "label": label,
                "confidence": float(score),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "is_anomaly": False,
            }
        )
    return detections


def _mark_anomaly(detections, anomalies):
    anomaly_boxes = {tuple(a["box"]) for a in anomalies}
    for det in detections:
        if tuple(det["box"]) in anomaly_boxes:
            det["is_anomaly"] = True


def get_color(class_id):
    # Tạo sẵn bảng màu sắc nét cho từng class_id
    colors = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
    ]
    return colors[class_id % len(colors)]


def _draw_detections(image, detections, line_width=2, font_size=1):
    vis = image.copy()
    # Chuyển đổi cỡ chữ giao diện (1-10) sang hệ scale của OpenCV
    font_scale = 0.5 + (font_size - 1) * 0.15
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = get_color(det["class_id"])
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, int(line_width))
        cv2.putText(
            vis,
            f"{det['label']} {det['confidence']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(line_width) - 1),
            cv2.LINE_AA,
        )
    return vis


def run_pcb_scan(model_path, image_path, output_dir, conf_thres=0.25, iou_thres=0.5, use_sahi=False, line_width=3, font_size=3):
    start_t = time.time()

    model_path = resolve_project_path(model_path)
    if not model_path or not os.path.exists(model_path):
        return None, "LOI: Khong tim thay model.", None, None
    if not image_path or not os.path.exists(image_path):
        return None, "LOI: Khong tim thay anh dau vao.", None, None

    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        return None, "LOI: Khong doc duoc anh.", None, None

    model = _get_model(model_path)
    raw_dets = infer_with_optional_slicing(model, image, conf=conf_thres, iou=iou_thres, use_sahi=use_sahi)
    class_names = _get_class_names(model)
    detections = _build_detections(raw_dets, class_names)

    # Tạm thời gán rỗng vì hàm detect_anomalies chưa được định nghĩa
    anomalies = []
    _mark_anomaly(detections, anomalies)
    vis = _draw_detections(image, detections, line_width=line_width, font_size=font_size)

    end_t = time.time()
    processing_time = end_t - start_t

    log_data_dir = os.path.join("/mnt/data_log", "LogData")
    try:
        os.makedirs(log_data_dir, exist_ok=True)
    except Exception:
        log_data_dir = os.path.join(output_dir, "LogData")
        os.makedirs(log_data_dir, exist_ok=True)

    now_dt = datetime.datetime.now()
    dt_str = now_dt.strftime("%Y%m%d_%H%M%S")

    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join(log_data_dir, f"data_{model_basename}.csv")
    file_exists = os.path.exists(csv_path)

    # Sử dụng trực tiếp class_names từ model theo yêu cầu
    defect_list = class_names

    detected_dict = {}
    for d in detections:
        label = d["label"]
        conf = d["confidence"]
        # Chỉ lưu lại độ tin cậy cao nhất nếu có nhiều lỗi cùng loại
        if label not in detected_dict or conf > detected_dict[label]:
            detected_dict[label] = conf

    has_defect = 0
    row_data = {}
    for defect in defect_list:
        if defect in detected_dict:
            row_data[defect] = 1
            row_data[f"{defect}_Conf"] = f"{detected_dict[defect]:.2f}"
            has_defect = 1
        else:
            row_data[defect] = 0
            row_data[f"{defect}_Conf"] = ""
            
    status = 1 if has_defect else 0

    status_folder = "NG" if status == 1 else "OK"
    log_img_dir = os.path.join("/mnt/data_log", "LogImage", status_folder)
    try:
        os.makedirs(log_img_dir, exist_ok=True)
    except Exception:
        log_img_dir = os.path.join(output_dir, "LogImage", status_folder)
        os.makedirs(log_img_dir, exist_ok=True)

    out_img_path = os.path.join(log_img_dir, f"{dt_str}.jpg")
    cv2.imwrite(out_img_path, vis)

    stt = 1
    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            stt = sum(1 for _ in f)

    headers = ["STT", "Datetime", "ProcessingTime"]
    for defect in defect_list:
        headers.append(defect)
        headers.append(f"{defect}_Conf")
    headers.append("Status")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        row = [stt, dt_str, f"{processing_time:.3f}"]
        for defect in defect_list:
            row.append(row_data[defect])
            row.append(row_data[f"{defect}_Conf"])
        row.append(status)
        writer.writerow(row)

    summary = (
        f"THANH CONG: Da quet anh\n"
        f"So loi detect: {len(detections)}\n"
        f"So anomaly candidate: {len(anomalies)}\n"
        f"Anh ket qua: {out_img_path}\n"
        f"Bao cao CSV: {csv_path}\n"
    )
    return out_img_path, summary, csv_path, ""
