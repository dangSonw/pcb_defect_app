import gradio as gr
import os
import cv2
import tempfile
import numpy as np
import json
import time
import threading
from core.inference_engine import run_pcb_scan
from core.camera_service import get_camera
from core.dataset_manager import load_system_config


# --- BỘ NHỚ ĐỆM ĐỒNG BỘ GIAO DIỆN TỪ XA ---
SHARED_STATE = {
    "has_new_capture": False,
    "has_new_scan": False,
    "capture_path": None,
    "scan_img": None,
    "scan_log": "",
    "scan_data": None,
    "plc_connected": False,
    "plc_vars": {
        "M1020": 0,
        "M169": 0,
        "M170": 0,
        "M171": 0,
    }
}

PLC_STATE = {
    "ip": "192.168.1.10",  # IP thực tế của PLC FX5U
    "port": 5009,           # Port giao thức MC (Type3E)
    "connected": False
}


def apply_plc_result_state(plc, has_defect):
    plc.batchwrite_bitunits("M169", [0])
    if has_defect:
        plc.batchwrite_bitunits("M171", [1])
        plc.batchwrite_bitunits("M170", [0])
    else:
        plc.batchwrite_bitunits("M170", [1])
        plc.batchwrite_bitunits("M171", [0])


def plc_worker():
    import pymcprotocol
    
    prev_m1020_state = False  # Biến trạng thái để phát hiện sườn lên
    plc = None
    last_connect_time = 0
    RECONNECT_DELAY = 3  # Giây chờ giữa mỗi lần thử kết nối lại

    while True:
        try:
            if not PLC_STATE["connected"]:
                current_time = time.time()
                if current_time - last_connect_time < RECONNECT_DELAY:
                    time.sleep(0.1)
                    continue
                    
                prev_m1020_state = False  # Reset trạng thái khi kết nối lại
                print(f"[PLC] Đang thử kết nối tới {PLC_STATE['ip']}:{PLC_STATE['port']}...")
                
                # Khởi tạo lại Type3E mỗi lần connect để đảm bảo tạo Socket hoàn toàn mới
                plc = pymcprotocol.Type3E()
                plc.setaccessopt(commtype="binary")
                if hasattr(plc, 'timer'):
                    plc.timer = 2
                
                plc.connect(PLC_STATE["ip"], PLC_STATE["port"])
                PLC_STATE["connected"] = True
                SHARED_STATE["plc_connected"] = True
                print(f"[PLC] Đã kết nối thành công tới {PLC_STATE['ip']}:{PLC_STATE['port']}")
                time.sleep(0.1) # Đợi PLC ổn định socket sau khi kết nối
                
            if not plc:
                continue
                
            # Đọc tín hiệu từ PLC (M1020)
            read_data = plc.batchread_bitunits("M1020", 1)
            current_m1020_state = read_data[0]
            SHARED_STATE["plc_vars"]["M1020"] = current_m1020_state
            
            # Cập nhật các biến khác để hiển thị lên UI
            read_m169x = plc.batchread_bitunits("M169", 3)
            SHARED_STATE["plc_vars"]["M169"] = read_m169x[0]
            SHARED_STATE["plc_vars"]["M170"] = read_m169x[1]
            SHARED_STATE["plc_vars"]["M171"] = read_m169x[2]
            
            # Phát hiện sườn lên (chuyển từ False -> True)
            if current_m1020_state and not prev_m1020_state:
                print("[PLC] Nhận lệnh chụp và rà quét AI (sườn lên M1020)")
                try:
                    # Bật M169 báo hiệu đang chụp và xử lý AI
                    plc.batchwrite_bitunits("M169", [1])
                    plc.batchwrite_bitunits("M170", [0])
                    plc.batchwrite_bitunits("M171", [0])
                    print("[PLC] Đã bật M169 (Đang xử lý)")
                    
                    cfg = load_system_config()
                    out_dir = cfg.get("output_dir", "outputs")
                    device = cfg.get("device", "CPU")
                    model_path = cfg.get("model_path", "")
                    
                    # 1. Thực hiện chụp ảnh
                    filepath, msg = capture_from_csi(out_dir)
                    if filepath and os.path.exists(filepath):
                        SHARED_STATE["capture_path"] = filepath
                        SHARED_STATE["has_new_capture"] = True
                        
                        # 2. Rà quét AI ngay lập tức với ảnh vừa chụp
                        res_img, log_text, log_data = start_inference(
                            sys_device=device, sys_model_path=model_path, sys_output_dir=out_dir,
                            webcam_image_path=filepath, conf_thres=0.25, line_width=2, font_size=1
                        )
                        if res_img:
                            SHARED_STATE["scan_img"] = res_img
                            SHARED_STATE["scan_log"] = log_text
                            SHARED_STATE["scan_data"] = log_data
                            SHARED_STATE["has_new_scan"] = True
                            
                            has_defect = "So loi detect: 0" not in log_text
                            apply_plc_result_state(plc, has_defect=has_defect)
                            print(f"[PLC] Quét hoàn tất. M169=0, {'M171=1' if has_defect else 'M170=1'}")
                        else:
                            print(f"[PLC] Cảnh báo: Lỗi nội bộ khi phân tích AI. Msg: {log_text}")
                            apply_plc_result_state(plc, has_defect=True)
                    else:
                        print(f"[PLC] Cảnh báo: Không thể chụp ảnh từ Camera. Msg: {msg}")
                        apply_plc_result_state(plc, has_defect=True)
                except Exception as proc_err:
                    print(f"[PLC] Lỗi trong quá trình xử lý: {proc_err}")
                finally:
                    # Tắt M169 khi hoàn thành (hoặc có lỗi)
                    try:
                        plc.batchwrite_bitunits("M169", [0])
                        print("[PLC] Đã tắt M169 (Hoàn tất xử lý)")
                    except Exception:
                        pass
            
            # Cập nhật trạng thái của M1020 cho lần lặp kế tiếp
            prev_m1020_state = current_m1020_state
                    
        except Exception as e:
            if PLC_STATE["connected"]:
                print(f"[PLC] Lỗi mất kết nối: {e}")
            else:
                print(f"[PLC] Lỗi kết nối tới {PLC_STATE['ip']}:{PLC_STATE['port']} - Chi tiết: {e}")
            PLC_STATE["connected"] = False
            SHARED_STATE["plc_connected"] = False
            last_connect_time = time.time()  # Cập nhật thời gian mất kết nối để delay reconnect
            prev_m1020_state = False  # Reset trạng thái khi có lỗi
            if plc is not None:
                try:
                    plc.close()  # Bắt buộc đóng socket bị lỗi
                except Exception:
                    pass
                plc = None
            
        time.sleep(0.2) # Giãn thời gian poll (200ms) để tránh PLC bị quá tải gây Broken Pipe

# Khởi chạy thread đọc PLC ngầm
plc_thread = threading.Thread(target=plc_worker, daemon=True)
plc_thread.start()

def _dbg(hypothesis_id, location, message, data):
    # Vô hiệu hoá I/O làm chậm
    pass

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

def capture_from_csi(sys_output_dir):
    """Chụp ảnh từ camera CSI và trả về đường dẫn"""
    # Dùng tempdir để lưu trên RAM ảo thay vì ghi rác ra ổ cứng outputs/
    output_dir = tempfile.gettempdir()
    try:
        camera = get_camera()
        if not camera.is_running:
            camera.start()
        
        filepath, msg = camera.capture(output_dir)
        if filepath and os.path.exists(filepath):
            return filepath, msg
        else:
            return None, msg
    except Exception as e:
        return None, f"Lỗi: {e}"

def start_inference(
    sys_device,
    sys_model_path,
    sys_output_dir,
    webcam_image_path,
    conf_thres,
    line_width=2,
    font_size=1,
):
    _ = sys_device  # device is already inferred by backend dependencies
    
    # Lấy output_dir từ config
    cfg = load_system_config()
    output_dir = cfg.get("output_dir", "outputs")

    in_image = _resolve_image_path(webcam_image_path)
    
    if not in_image or not os.path.exists(in_image):
        error_msg = (
            "❌ LỖI: Chưa có ảnh để rà quét!\n\n"
            "HƯỚNG DẪN:\n"
            "1. Sử dụng Camera CSI: Nhấn '🎥 Chụp từ Camera CSI'\n"
            "2. Upload ảnh: Chọn tab '📤 Upload từ máy tính' và tải ảnh lên\n"
            "3. Sau đó nhấn '▶️ BẮT ĐẦU RÀ QUÉT'"
        )
        return None, error_msg, []

    result_img, log_text, csv_path, _ = run_pcb_scan(
        model_path=sys_model_path,
        image_path=in_image,
        output_dir=output_dir,
        conf_thres=conf_thres,
        line_width=line_width,
        font_size=font_size,
    )

    # Tạo bản sao ảnh kết quả vào thư mục Temp để Gradio có quyền truy cập và hiển thị
    display_img = None
    if result_img and os.path.exists(result_img):
        import shutil
        tmp_dir = tempfile.gettempdir()
        display_img = os.path.join(tmp_dir, f"display_{int(time.time())}.jpg")
        shutil.copy(result_img, display_img)

    log_data = []
    try:
        import pandas as pd
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log_data = df.tail(10).iloc[::-1]
    except ImportError:
        try:
            import csv
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    rows = list(csv.reader(f))
                    if len(rows) > 1:
                        log_data = [rows[0]] + rows[1:][::-1][:10]
                    elif len(rows) == 1:
                        log_data = [rows[0]]
        except Exception:
            pass
    except Exception:
        pass

    return display_img, log_text, log_data

def save_uploaded_file(uploaded_file, output_dir):
    """Lưu file upload từ client vào output_dir"""
    if uploaded_file is None:
        return None, "Chưa có file nào được chọn"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"upload_{int(time.time())}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Nếu là PIL Image hoặc numpy array
        if isinstance(uploaded_file, np.ndarray):
            cv2.imwrite(filepath, uploaded_file)
        else:
            # Nếu là file path
            if isinstance(uploaded_file, dict) and 'name' in uploaded_file:
                filepath = uploaded_file['name']
            else:
                filepath = str(uploaded_file)
        
        return filepath, f"✓ File đã tải: {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Lỗi tải file: {e}"

def render(sys_device, sys_model_path, sys_output_dir, camera_available=False, app=None):
    with gr.Row():
        # CỘT 1: THÔNG SỐ RÀ QUÉT
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 Trạng thái PLC")
            with gr.Row():
                ui_plc_status = gr.Textbox(label=f"Kết nối ({PLC_STATE['ip']})", value="🔴 Mất kết nối", interactive=False, scale=2)
                ui_m1020 = gr.Textbox(label="M1020 (Trigger)", value="⚫ 0", interactive=False, scale=1)
            with gr.Row():
                ui_m169 = gr.Textbox(label="M169 (Busy)", value="⚫ 0", interactive=False, scale=1)
                ui_m170 = gr.Textbox(label="M170 (OK)", value="⚫ 0", interactive=False, scale=1)
                ui_m171 = gr.Textbox(label="M171 (NG)", value="⚫ 0", interactive=False, scale=1)
            
            gr.Markdown("### 📸 Chọn ảnh để quét")
            
            # Hiển thị cảnh báo nếu camera không khả dụng
            if not camera_available:
                gr.Markdown("⚠️ **Camera CSI không khả dụng** - Sử dụng Upload ảnh")
            
            # Tab cho 2 phương thức input
            with gr.Tabs():
                with gr.TabItem("🎥 Camera CSI Jetson"):
                    gr.Markdown("Chụp trực tiếp từ camera CSI trên Jetson Nano")
                    ui_csi_image = gr.Image(label="Ảnh vừa chụp từ Camera", type="filepath", interactive=False)
                    csi_capture_btn = gr.Button(
                        "🎥 Chụp từ Camera CSI" if camera_available else "❌ Camera không khả dụng",
                        variant="secondary", 
                        size="lg",
                        interactive=camera_available
                    )
                    ui_csi_status = gr.Textbox(
                        label="Trạng thái",
                        interactive=False,
                        value="Camera không khả dụng" if not camera_available else "Sẵn sàng"
                    )
                
                with gr.TabItem("📤 Upload từ máy tính"):
                    gr.Markdown("Tải ảnh lên từ máy tính của bạn")
                    ui_upload_image = gr.Image(type="filepath", sources=["upload"], label="Chọn ảnh PCB", scale=1)
                    ui_upload_status = gr.Textbox(label="Trạng thái upload", interactive=False, value="Chờ upload...")
                    ui_uploaded_image = gr.State(None)
            
            gr.Markdown("### ⚙️ Cấu hình rà quét")
            ui_conf_thres = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Ngưỡng tự tin (Confidence Threshold)")
            
            with gr.Row():
                ui_line_width = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Độ dày Bounding Box")
                ui_font_size = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Cỡ chữ (Font Size)")
            
            run_btn = gr.Button("▶️ BẮT ĐẦU RÀ QUÉT", variant="primary", size="lg")
            ui_log_output = gr.Textbox(label="📋 Báo cáo", lines=8, interactive=False)

        # CỘT 2: HIỂN THỊ KẾT QUẢ TRỰC QUAN
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Màn hình QA/QC Trực quan")
            ui_result_img = gr.Image(label="Ảnh kết quả (Đã đánh dấu lỗi)", type="filepath")
            
            gr.Markdown("### 🗄️ Dữ liệu Log (10 mẫu mới nhất)")
            ui_log_table = gr.Dataframe(interactive=False, wrap=True)
            
    # Sự kiện: Chụp từ Camera CSI
    csi_capture_btn.click(
        fn=capture_from_csi,
        inputs=[sys_output_dir],
        outputs=[ui_csi_image, ui_csi_status],
    ).then(
        fn=lambda: gr.update(value="✓ Chụp thành công! Nhấn 'BẮT ĐẦU RÀ QUÉT' để phân tích."),
        outputs=[ui_csi_status]
    )
    
    # Sự kiện: Upload file
    ui_upload_image.change(
        fn=save_uploaded_file,
        inputs=[ui_upload_image, sys_output_dir],
        outputs=[ui_uploaded_image, ui_upload_status]
    )

    # Chạy inference với ảnh được chọn (CSI hoặc Upload)
    def run_with_selected_image(sys_device, sys_model_path, sys_output_dir, csi_image_path, uploaded_image_path, conf_thres, line_width, font_size):
        # Xác định ảnh mới nhất được chọn giữa CSI và Upload
        image_input = None
        if csi_image_path and uploaded_image_path:
            try:
                if os.path.getmtime(csi_image_path) > os.path.getmtime(uploaded_image_path):
                    image_input = csi_image_path
                else:
                    image_input = uploaded_image_path
            except Exception:
                image_input = csi_image_path
        else:
            image_input = csi_image_path if csi_image_path else uploaded_image_path
        return start_inference(sys_device, sys_model_path, sys_output_dir, image_input, conf_thres, line_width, font_size)

    run_btn.click(
        fn=run_with_selected_image,
        inputs=[
            sys_device,
            sys_model_path,
            sys_output_dir,
            ui_csi_image,
            ui_uploaded_image,
            ui_conf_thres,
            ui_line_width,
            ui_font_size,
        ],
        outputs=[ui_result_img, ui_log_output, ui_log_table],
    )

    if app is not None:
        timer = gr.Timer(0.2)
        
        def sync_ui():
            plc_str = "🟢 Đã kết nối" if SHARED_STATE["plc_connected"] else "🔴 Mất kết nối"
            m1020_str = "🟢 1" if SHARED_STATE["plc_vars"]["M1020"] else "⚫ 0"
            m169_str = "🟢 1" if SHARED_STATE["plc_vars"]["M169"] else "⚫ 0"
            m170_str = "🟢 1" if SHARED_STATE["plc_vars"]["M170"] else "⚫ 0"
            m171_str = "🔴 1" if SHARED_STATE["plc_vars"]["M171"] else "⚫ 0"
            
            updates = [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=plc_str), gr.update(value=m1020_str), gr.update(value=m169_str), gr.update(value=m170_str), gr.update(value=m171_str)]
            if SHARED_STATE["has_new_capture"]:
                SHARED_STATE["has_new_capture"] = False
                updates[0] = SHARED_STATE["capture_path"]
                updates[1] = "✓ Đã chụp từ PLC"
            if SHARED_STATE["has_new_scan"]:
                SHARED_STATE["has_new_scan"] = False
                updates[2] = SHARED_STATE["scan_img"]
                updates[3] = SHARED_STATE["scan_log"]
                updates[4] = SHARED_STATE["scan_data"]
            return tuple(updates)
            
        timer.tick(
            fn=sync_ui,
            inputs=[],
            outputs=[ui_csi_image, ui_csi_status, ui_result_img, ui_log_output, ui_log_table, ui_plc_status, ui_m1020, ui_m169, ui_m170, ui_m171]
        )