import cv2
import numpy as np


def infer_with_optional_slicing(model, image_bgr, conf=0.25, iou=0.5, use_sahi=False):
    """
    Run prediction with optional tiled slicing.
    Returns a flat list: [x1,y1,x2,y2,cls_id,score]
    """
    if image_bgr is None:
        return []
    if not use_sahi:
        return _infer_one(model, image_bgr, 0, 0, conf, iou)

    h, w = image_bgr.shape[:2]
    tile = 640
    overlap = 0.2
    step = int(tile * (1.0 - overlap))
    detections = []
    for y in range(0, h, max(1, step)):
        for x in range(0, w, max(1, step)):
            crop = image_bgr[y:min(y + tile, h), x:min(x + tile, w)]
            if crop.size == 0:
                continue
            dets = _infer_one(model, crop, x, y, conf, iou)
            detections.extend(dets)
    return _nms_merge(detections, iou_threshold=iou)


def _infer_one(model, image_bgr, offset_x, offset_y, conf, iou):
    results = model.predict(source=image_bgr, conf=float(conf), iou=float(iou), verbose=False)
    if not results:
        return []
    out = []
    res = results[0]
    if res.boxes is None:
        return out
    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
    cls_ids = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else np.array(res.boxes.cls)
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        out.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, int(cls_ids[i]), float(confs[i])])
    return out


def _nms_merge(detections, iou_threshold=0.5):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[5], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_threshold]
    return keep


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0
# vision_slicer.py
# Tích hợp SAHI để rà quét ảnh PCB bằng cửa sổ trượt
