import gradio as gr
import os
import cv2
import tempfile
import numpy as np
import json
import time
from core.inference_engine import run_pcb_scan


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

def _resolve_path(path_value):
    if isinstance(path_value, dict):
        return path_value.get("path") or path_value.get("video") or ""
    if isinstance(path_value, (list, tuple)) and path_value:
        return str(path_value[0])
    if isinstance(path_value, str):
        return path_value
    if hasattr(path_value, "name"):
        return path_value.name
    return ""


def _resolve_image_path(image_value):
    path = _resolve_path(image_value)
    # region agent log
    _dbg("H1", "ui/tab_run.py:_resolve_image_path", "resolve image input type", {"input_type": str(type(image_value)), "resolved_path": path})
    # endregion
    if path and os.path.exists(path):
        return path
    if isinstance(image_value, np.ndarray):
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, "pcb_camera_snapshot.jpg")
        # Gradio webcam ndarray is RGB; OpenCV expects BGR.
        bgr = cv2.cvtColor(image_value, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, bgr)
        return tmp_path

    # Thử chụp tự động bằng OpenCV nếu chưa có ảnh từ UI (cho trường hợp người dùng ẩn Camera)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, "pcb_auto_snapshot.jpg")
                cv2.imwrite(tmp_path, frame)
                return tmp_path
    except Exception:
        pass
    return ""

def start_inference(
    sys_device,
    sys_model_path,
    sys_output_dir,
    webcam_image_path,
    use_sahi,
    conf_thres,
    iou_thres,
):
    _ = sys_device  # device is already inferred by backend dependencies
    output_dir = sys_output_dir or "outputs"
    # region agent log
    _dbg("H2", "ui/tab_run.py:start_inference", "inference entry", {"model_path": sys_model_path, "output_dir": output_dir})
    # endregion

    in_image = _resolve_image_path(webcam_image_path)
    # region agent log
    _dbg("H4", "ui/tab_run.py:start_inference", "image resolved path", {"image_path": in_image, "image_exists": os.path.exists(in_image) if in_image else False})
    # endregion
    
    if not in_image or not os.path.exists(in_image):
        error_msg = (
            "LỖI: Chưa có ảnh để rà quét!\n\n"
            "HƯỚNG DẪN:\n"
            "- Cách 1: Khi Camera đang bật, bạn PHẢI nhấn vào biểu tượng máy ảnh (📸) trên khung hình để chụp một tấm ảnh tĩnh, sau đó mới nhấn 'BẮT ĐẦU RÀ QUÉT'.\n"
            "- Cách 2: Tắt tùy chọn 'Bật/Tắt hiển thị Camera' và nhấn 'BẮT ĐẦU RÀ QUÉT', hệ thống sẽ tự động chụp ngầm từ webcam."
        )
        return None, error_msg

    result_img, log_text, _, _ = run_pcb_scan(
        model_path=sys_model_path,
        image_path=in_image,
        output_dir=output_dir,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        use_sahi=use_sahi,
    )
    # region agent log
    _dbg("H4", "ui/tab_run.py:start_inference", "image result", {"result_img": result_img, "log_text": log_text})
    # endregion
    return result_img, log_text

def render(sys_device, sys_model_path, sys_output_dir):
    with gr.Row():
        # CỘT 1: THÔNG SỐ RÀ QUÉT
        with gr.Column(scale=1):
            gr.Markdown("### Chụp ảnh & Cấu hình")
            ui_show_cam = gr.Checkbox(value=True, label="Bật/Tắt hiển thị Camera")
            ui_cam_input = gr.Image(type="filepath", sources=["webcam"], label="Chụp ảnh PCB từ Camera", visible=True)
                
            ui_use_sahi = gr.Checkbox(value=True, label="Kích hoạt SAHI (Cắt lớp ảnh rà vi khuyết tật)")
            ui_conf_thres = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Ngưỡng tự tin (Confidence Threshold)")
            ui_iou_thres = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Nguong IoU (NMS)")
            
            run_btn = gr.Button("BẮT ĐẦU RÀ QUÉT", variant="primary")
            ui_log_output = gr.Textbox(label="Báo cáo dạng Text", lines=5, interactive=False)

        # CỘT 2: HIỂN THỊ KẾT QUẢ TRỰC QUAN
        with gr.Column(scale=2):
            gr.Markdown("### Màn hình QA/QC Trực quan")
            ui_result_img = gr.Image(label="Ảnh kết quả (Đã đánh dấu lỗi)", type="filepath")
            
    ui_show_cam.change(
        fn=lambda show: gr.update(visible=show),
        inputs=[ui_show_cam],
        outputs=[ui_cam_input]
    )

    run_btn.click(
        fn=start_inference,
        inputs=[
            sys_device,
            sys_model_path,
            sys_output_dir,
            ui_cam_input,
            ui_use_sahi,
            ui_conf_thres,
            ui_iou_thres,
        ],
        outputs=[ui_result_img, ui_log_output]
    )