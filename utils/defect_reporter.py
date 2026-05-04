import csv
import json
import os
from datetime import datetime


def save_defect_reports(output_dir, image_path, detections, anomalies):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{stem}_{ts}_defects.csv")
    json_path = os.path.join(output_dir, f"{stem}_{ts}_defects.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "confidence", "x1", "y1", "x2", "y2", "is_anomaly"],
        )
        writer.writeheader()
        for det in detections:
            box = det.get("box", [0, 0, 0, 0])
            writer.writerow(
                {
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.0),
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "is_anomaly": det.get("is_anomaly", False),
                }
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image": image_path, "detections": detections, "anomalies": anomalies}, f, ensure_ascii=False, indent=2)

    return csv_path, json_path
