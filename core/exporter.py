import os
import time
from ultralytics import YOLO

def export_model(model_path, export_format, use_half, use_int8, imgsz, device):
    try:
        if not model_path or not os.path.exists(model_path):
            return "LOI: Khong tim thay file model goc.", model_path
        
        if not str(model_path).endswith('.pt'):
            return "LOI: Chi ho tro export tu file .pt", model_path

        format_code = "engine" if export_format == "TensorRT (.engine)" else "onnx"
        export_device = "0" if ("GPU" in device or "cuda" in str(device).lower()) else "cpu"
        
        if format_code == "engine" and export_device == "cpu":
            return "LOI TUONG THICH: TensorRT can GPU/CUDA.", model_path
        if format_code == "engine":
            try:
                import tensorrt  # noqa: F401
            except Exception:
                return "LOI TUONG THICH: Chua co TensorRT runtime trong moi truong.", model_path

        started = time.time()
        model = YOLO(model_path)
        exported_path = model.export(
            format=format_code,
            half=use_half,
            int8=use_int8,
            imgsz=int(imgsz),
            device=export_device,
            simplify=True if format_code == "onnx" else False
        )
        elapsed = time.time() - started
        exported_path = str(exported_path)

        return (
            "THANH CONG: Da bien dich model\n"
            f"Dinh dang: {format_code}\n"
            f"Output: {exported_path}\n"
            f"Thoi gian: {elapsed:.2f}s",
            exported_path,
        )

    except Exception as e:
        return f"LOI BIEN DICH:\n{str(e)}", model_path