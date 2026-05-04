def detect_anomalies(detections, class_names, anomaly_conf_threshold=0.35):
    """Danh dau cac detect co do tin cay thap la anomaly candidate."""
    _ = class_names
    anomalies = []
    for det in detections:
        score = float(det.get("confidence", 0.0))
        if score < float(anomaly_conf_threshold):
            anomalies.append(
                {
                    "type": "low_confidence",
                    "label": det.get("label", "unknown"),
                    "confidence": score,
                    "box": det.get("box", [0, 0, 0, 0]),
                }
            )
    return anomalies
