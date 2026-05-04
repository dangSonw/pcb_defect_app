import os
import cv2
import time
import json
from ultralytics import YOLO

from core.anomaly_engine import detect_anomalies
from utils.defect_reporter import save_defect_reports
from utils.vision_slicer import infer_with_optional_slicing


def _dbg(hypothesis_id, location, message, data):
    # region agent log
    try:
        payload = {
            "sessionId": "1005ba",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-1005ba.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


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


def _draw_detections(image, detections):
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 0, 255) if det["is_anomaly"] else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{det['label']} {det['confidence']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def run_pcb_scan(model_path, image_path, output_dir, conf_thres=0.25, iou_thres=0.5, use_sahi=False):
    # region agent log
    _dbg("H5", "core/inference_engine.py:run_pcb_scan", "entry", {"model_path": model_path, "image_path": image_path, "output_dir": output_dir, "use_sahi": use_sahi})
    # endregion
    if not model_path or not os.path.exists(model_path):
        return None, "LOI: Khong tim thay model.", None, None
    if not image_path or not os.path.exists(image_path):
        return None, "LOI: Khong tim thay anh dau vao.", None, None

    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        return None, "LOI: Khong doc duoc anh.", None, None

    model = YOLO(model_path)
    raw_dets = infer_with_optional_slicing(model, image, conf=conf_thres, iou=iou_thres, use_sahi=use_sahi)
    class_names = _get_class_names(model)
    detections = _build_detections(raw_dets, class_names)

    anomalies = detect_anomalies(detections, class_names)
    _mark_anomaly(detections, anomalies)
    vis = _draw_detections(image, detections)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_img_path, vis)
    csv_path, json_path = save_defect_reports(output_dir, image_path, detections, anomalies)
    summary = (
        f"THANH CONG: Da quet anh\n"
        f"So loi detect: {len(detections)}\n"
        f"So anomaly candidate: {len(anomalies)}\n"
        f"Anh ket qua: {out_img_path}\n"
        f"Bao cao CSV: {csv_path}\n"
        f"Bao cao JSON: {json_path}"
    )
    # region agent log
    _dbg("H5", "core/inference_engine.py:run_pcb_scan", "exit", {"out_img_path": out_img_path, "out_exists": os.path.exists(out_img_path), "detections": len(detections)})
    # endregion
    return out_img_path, summary, csv_path, json_path
