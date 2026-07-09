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
    },
    # AI Flow step: 0=idle, 1=acquiring, 2=preprocessing, 3=inference, 4=output
    "ai_flow_step": 0,
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
                
                # Khởi tạo prev_m1020_state bằng giá trị hiện tại của PLC để tránh việc nhận nhầm sườn lên khi mới kết nối
                try:
                    init_data = plc.batchread_bitunits("M1020", 1)
                    prev_m1020_state = init_data[0]
                except Exception:
                    prev_m1020_state = False
                
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
                    SHARED_STATE["ai_flow_step"] = 1
                    filepath, msg = capture_from_csi(out_dir)
                    if filepath and os.path.exists(filepath):
                        SHARED_STATE["capture_path"] = filepath
                        SHARED_STATE["has_new_capture"] = True
                        
                        # 2. Rà quét AI ngay lập tức với ảnh vừa chụp
                        SHARED_STATE["ai_flow_step"] = 2
                        time.sleep(0.3)  # brief pause so UI shows pre-processing
                        SHARED_STATE["ai_flow_step"] = 3
                        res_img, log_text, log_data = start_inference(
                            sys_device=device, sys_model_path=model_path, sys_output_dir=out_dir,
                            webcam_image_path=filepath, conf_thres=0.25, line_width=2, font_size=1
                        )
                        SHARED_STATE["ai_flow_step"] = 4
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
                    time.sleep(2.0)  # hold output step 4 for 2s before resetting
                    SHARED_STATE["ai_flow_step"] = 0
                    
                    # Đọc lại trạng thái thực tế của M1020 sau thời gian xử lý và sleep lâu, tránh lệch sườn do delay
                    try:
                        read_data = plc.batchread_bitunits("M1020", 1)
                        current_m1020_state = read_data[0]
                        SHARED_STATE["plc_vars"]["M1020"] = current_m1020_state
                        print(f"[PLC] Đọc lại trạng thái M1020 sau xử lý: {current_m1020_state}")
                    except Exception as read_err:
                        print(f"[PLC] Lỗi đọc lại M1020 sau xử lý: {read_err}")
            
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
    """Lưu file upload từ client vào output_dir (luôn copy để tránh temp-path expired)."""
    if uploaded_file is None:
        return None, "Chưa có file nào được chọn"

    try:
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        filename = f"upload_{int(time.time())}.jpg"
        dest = os.path.join(output_dir, filename)

        if isinstance(uploaded_file, np.ndarray):
            # numpy array (webcam frame)
            cv2.imwrite(dest, uploaded_file)
        else:
            # Gradio trả về string path (type="filepath")
            src = (
                uploaded_file["name"]
                if isinstance(uploaded_file, dict) and "name" in uploaded_file
                else str(uploaded_file)
            )
            if src and os.path.exists(src):
                shutil.copy(src, dest)   # copy sang dest bền vững
            else:
                # fallback: dùng path gốc nếu không copy được
                dest = src

        return dest, f"✓ File đã tải: {os.path.basename(dest)}"
    except Exception as e:
        return None, f"Lỗi tải file: {e}"



# ---------------------------------------------------------------------------
#  Helper: build AI flow HTML  (Tab 5)
#  Layout  : 2x2 cyclic  (light mode)
#
#   [Data Acquisition] --(arr_r)--> [Pre-processing]
#          ^                               |
#        (arr_u)                        (arr_d)
#          |                               v
#   [Output Handling] <--(arr_l)-- [Model Inference]
#
#  All sizing is percentage-based / 1fr grid so the component never overflows.
# ---------------------------------------------------------------------------
def _build_flow_html(active_step: int) -> str:
    """
    active_step:
      0 = idle
      1 = Data Acquisition  (top-left)
      2 = Pre-processing     (top-right)
      3 = Model Inference    (bottom-right)
      4 = Output Handling    (bottom-left)
    """
    plc_info  = "PLC: Mat ket noi" if not SHARED_STATE["plc_connected"] else "PLC: Da ket noi"
    plc_icon  = "\U0001f534" if not SHARED_STATE["plc_connected"] else "\U0001f7e2"
    plc_color = "#dc2626"    if not SHARED_STATE["plc_connected"] else "#16a34a"

    status_map = {
        0: "\u23f8\ufe0f Cho lenh kich hoat...",
        1: "\U0001f4f7 Dang thu du lieu tu Camera...",
        2: "\U0001f527 Tien xu ly anh...",
        3: "\U0001f9e0 Model AI dang suy luan...",
        4: "\u2705 Xu ly hoan tat \u2014 Hien thi & Luu tru",
    }
    status_label = status_map.get(active_step, "")

    # ------------------------------------------------------------------ helpers
    def _state(step):
        """Return (bg, border, shadow, label_color, sub_color, anim, badge_html)"""
        active = step == active_step
        done   = active_step > 0 and step < active_step
        if active:
            return (
                "#dbeafe", "#3b82f6",
                "0 0 0 3px #bfdbfe, 0 4px 14px #3b82f640",
                "#1e40af", "#2563eb",
                "animation:fp5-pulse 1.2s ease-in-out infinite;",
                '<div style="position:absolute;top:-11px;right:-4px;'
                'background:#3b82f6;color:#fff;border-radius:999px;'
                'font-size:0.58rem;font-weight:700;padding:2px 7px;'
                'white-space:nowrap;line-height:1.5;">ACTIVE</div>',
            )
        if done:
            return (
                "#dcfce7", "#22c55e",
                "0 2px 8px #22c55e22",
                "#166534", "#15803d",
                "",
                '<div style="position:absolute;top:-11px;right:-4px;'
                'background:#22c55e;color:#fff;border-radius:999px;'
                'font-size:0.58rem;font-weight:700;padding:2px 7px;'
                'white-space:nowrap;line-height:1.5;">DONE</div>',
            )
        return (
            "#f8fafc", "#cbd5e1",
            "0 2px 6px rgba(0,0,0,.05)",
            "#334155", "#94a3b8",
            "", "",
        )

    def box(step, label, sub):
        bg, bd, sh, lc, sc, an, badge = _state(step)
        return (
            '<div style="position:relative;box-sizing:border-box;min-width:0;width:100%;'
            f'background:{bg};border:2px solid {bd};border-radius:14px;'
            f'padding:12px 8px;display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;box-shadow:{sh};'
            f'transition:background .35s,border .35s,box-shadow .35s;{an}">'
            + badge +
            f'<div style="font-weight:800;font-size:0.82rem;color:{lc};'
            f'text-align:center;line-height:1.35;word-break:break-word;">{label}</div>'
            f'<div style="font-size:0.64rem;color:{sc};margin-top:4px;'
            f'text-align:center;line-height:1.3;word-break:break-word;">{sub}</div>'
            '</div>'
        )

    def arr_color(from_step):
        if active_step > from_step: return "#22c55e"
        if active_step == from_step: return "#3b82f6"
        return "#d1d5db"

    def flowing(from_step):
        return active_step == from_step

    def harr(from_step, flip=False):
        c   = arr_color(from_step)
        flo = flowing(from_step)
        shine = (
            '<div class="fp5-flow" style="position:absolute;top:0;left:0;'
            'height:100%;width:45%;'
            'background:linear-gradient(90deg,transparent,rgba(255,255,255,.65),transparent);"></div>'
            if flo else ""
        )
        flip_s = "transform:scaleX(-1);" if flip else ""
        return (
            f'<div style="display:flex;align-items:center;width:100%;box-sizing:border-box;{flip_s}">'
            f'<div style="position:relative;flex:1;min-width:0;height:6px;border-radius:3px;'
            f'background:{c};overflow:hidden;transition:background .35s;">{shine}</div>'
            f'<div style="width:0;height:0;border-top:7px solid transparent;'
            f'border-bottom:7px solid transparent;border-left:12px solid {c};'
            f'flex-shrink:0;transition:border-color .35s;margin-left:-1px;"></div>'
            '</div>'
        )

    def varr(from_step, flip=False):
        c   = arr_color(from_step)
        flo = flowing(from_step)
        shine = (
            '<div class="fp5-flow-v" style="position:absolute;left:0;top:0;'
            'width:100%;height:45%;'
            'background:linear-gradient(180deg,transparent,rgba(255,255,255,.65),transparent);"></div>'
            if flo else ""
        )
        flip_s = "transform:scaleY(-1);" if flip else ""
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'box-sizing:border-box;height:100%;{flip_s}">'
            f'<div style="position:relative;width:6px;flex:1;border-radius:3px;'
            f'background:{c};overflow:hidden;transition:background .35s;">{shine}</div>'
            f'<div style="width:0;height:0;border-left:7px solid transparent;'
            f'border-right:7px solid transparent;border-top:12px solid {c};'
            f'flex-shrink:0;transition:border-color .35s;margin-top:-1px;"></div>'
            '</div>'
        )

    b1 = box(1, "Data<br>Acquisition",  "Camera &rarr; Frame")
    b2 = box(2, "Pre-<br>processing",   "Resize &middot; Normalize")
    b3 = box(3, "Model<br>Inference",   "YOLO &middot; ONNX &middot; TRT")
    b4 = box(4, "Output<br>Handling",   "UI &middot; CSV &middot; PLC")

    ar = harr(1)           # right  top
    ad = varr(2)           # down   right
    al = harr(3, flip=True)  # left   bottom
    au = varr(4, flip=True)  # up     left

    return (
        "<style>"
        ".fp5-wrap *{box-sizing:border-box;}"
        "@keyframes fp5-pulse{"
        "0%,100%{box-shadow:0 0 0 3px #bfdbfe,0 4px 16px #3b82f640;}"
        "50%{box-shadow:0 0 0 6px #93c5fd,0 4px 24px #3b82f680;}}"
        "@keyframes fp5-flow{"
        "0%{transform:translateX(-100%);}100%{transform:translateX(250%);}}"
        "@keyframes fp5-flow-v{"
        "0%{transform:translateY(-100%);}100%{transform:translateY(250%);}}"
        ".fp5-flow{animation:fp5-flow 1s linear infinite;}"
        ".fp5-flow-v{animation:fp5-flow-v 1s linear infinite;}"
        "</style>"

        # outer card — 100% width, overflow:hidden, no fixed pixel widths
        '<div class="fp5-wrap" style="'
        "width:100%;max-width:100%;box-sizing:border-box;overflow:hidden;"
        "background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;"
        "padding:16px 14px 12px;"
        "font-family:'Inter',system-ui,-apple-system,sans-serif;"
        'box-shadow:0 2px 12px rgba(0,0,0,.06);">'

        # header row
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:6px;flex-wrap:wrap;min-width:0;margin-bottom:4px;">'
        '<span style="color:#1e293b;font-size:0.88rem;font-weight:700;min-width:0;">'
        "\U0001f916 Quy trinh xu ly AI</span>"
        f'<span style="color:{plc_color};font-size:0.7rem;font-weight:600;'
        f'white-space:nowrap;flex-shrink:0;">{plc_icon} {plc_info}</span>'
        "</div>"

        # status line
        f'<div style="color:#3b82f6;font-size:0.76rem;font-weight:600;'
        f'min-height:1.1em;margin-bottom:12px;word-break:break-word;">{status_label}</div>'

        # ---- CSS Grid 3-col x 3-row, all sizes relative ----
        # columns: 1fr  |  clamp(36px,8%,48px)  |  1fr
        # rows:    auto |  clamp(36px,8vw,48px)  |  auto
        '<div style="'
        "display:grid;"
        "grid-template-columns:1fr clamp(36px,8%,48px) 1fr;"
        "grid-template-rows:auto clamp(36px,8vw,48px) auto;"
        "width:100%;min-width:0;"
        '">'

        # row 1
        f'<div style="min-width:0;padding:4px 0;">{b1}</div>'
        f'<div style="display:flex;align-items:center;justify-content:center;padding:0 3px;min-width:0;">{ar}</div>'
        f'<div style="min-width:0;padding:4px 0;">{b2}</div>'

        # row 2  (vertical arrows aligned to inner edges of their box)
        f'<div style="display:flex;align-items:stretch;justify-content:flex-start;'
        f'padding:3px 0 3px clamp(12px,15%,22px);min-width:0;">{au}</div>'
        '<div></div>'
        f'<div style="display:flex;align-items:stretch;justify-content:flex-end;'
        f'padding:3px clamp(12px,15%,22px) 3px 0;min-width:0;">{ad}</div>'

        # row 3
        f'<div style="min-width:0;padding:4px 0;">{b4}</div>'
        f'<div style="display:flex;align-items:center;justify-content:center;padding:0 3px;min-width:0;">{al}</div>'
        f'<div style="min-width:0;padding:4px 0;">{b3}</div>'

        '</div>'   # end grid

        # legend
        '<div style="display:flex;gap:10px;margin-top:12px;padding-top:10px;'
        'border-top:1px solid #f1f5f9;flex-wrap:wrap;">'
        '<span style="display:flex;align-items:center;gap:4px;font-size:0.67rem;color:#64748b;">'
        '<span style="width:10px;height:10px;border-radius:3px;flex-shrink:0;display:inline-block;'
        'background:#f8fafc;border:1.5px solid #cbd5e1;"></span>Cho</span>'
        '<span style="display:flex;align-items:center;gap:4px;font-size:0.67rem;color:#64748b;">'
        '<span style="width:10px;height:10px;border-radius:3px;flex-shrink:0;display:inline-block;'
        'background:#dbeafe;border:1.5px solid #3b82f6;"></span>Dang xu ly</span>'
        '<span style="display:flex;align-items:center;gap:4px;font-size:0.67rem;color:#64748b;">'
        '<span style="width:10px;height:10px;border-radius:3px;flex-shrink:0;display:inline-block;'
        'background:#dcfce7;border:1.5px solid #22c55e;"></span>Hoan thanh</span>'
        '</div>'

        '</div>'   # end card
    )



def render(sys_device, sys_model_path, sys_output_dir, camera_available=False, app=None):
    with gr.Row():
        # CỘT 1: THÔNG SỐ RÀ QUÉT
        with gr.Column(scale=1):
            # ── AI FLOW (replaces PLC status panel) ──────────────
            ui_ai_flow = gr.HTML(
                value=_build_flow_html(0),
                label="Quy trình xử lý AI"
            )

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
            
    gr.Markdown("---")
    gr.Markdown("### 🔌 Trạng thái các Bit PLC")
    with gr.Row():
        ui_plc_status = gr.Textbox(label=f"Kết nối ({PLC_STATE['ip']})", value="🔴 Mất kết nối", interactive=False, scale=2)
        ui_m1020 = gr.Textbox(label="M1020 (Trigger)", value="⚫ 0", interactive=False, scale=1)
        ui_m169 = gr.Textbox(label="M169 (Busy)", value="⚫ 0", interactive=False, scale=1)
        ui_m170 = gr.Textbox(label="M170 (OK)", value="⚫ 0", interactive=False, scale=1)
        ui_m171 = gr.Textbox(label="M171 (NG)", value="⚫ 0", interactive=False, scale=1)

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

    # Chạy inference với ảnh được chọn (CSI hoặc Upload) — cũng cập nhật AI flow

    def run_with_selected_image(sys_device, sys_model_path, sys_output_dir,
                                csi_image_path, uploaded_image_path,
                                conf_thres, line_width, font_size):
        """
        Generator: yields (flow_html, result_img, log_text, log_data) at each step
        so Gradio streams each update to the UI immediately — zero artificial delay.
        """
        # ── STEP 1: Data Acquisition ────────────────────────────────────────
        # Light up block INSTANTLY the moment acquisition begins
        SHARED_STATE["ai_flow_step"] = 1
        yield _build_flow_html(1), gr.update(), "📷 Đang thu dữ liệu từ Camera...", gr.update()

        # Actual work: validate & select image source
        csi_ok    = bool(csi_image_path    and os.path.exists(str(csi_image_path)))
        upload_ok = bool(uploaded_image_path and os.path.exists(str(uploaded_image_path)))
        if csi_ok and upload_ok:
            try:
                image_input = (
                    str(csi_image_path)
                    if os.path.getmtime(str(csi_image_path)) >
                       os.path.getmtime(str(uploaded_image_path))
                    else str(uploaded_image_path)
                )
            except Exception:
                image_input = str(csi_image_path)
        elif csi_ok:
            image_input = str(csi_image_path)
        elif upload_ok:
            image_input = str(uploaded_image_path)
        else:
            image_input = None

        # ── STEP 2: Pre-processing ──────────────────────────────────────────
        SHARED_STATE["ai_flow_step"] = 2
        yield _build_flow_html(2), gr.update(), "🔧 Tiền xử lý ảnh (resize · normalize)...", gr.update()

        # ── STEP 3: Model Inference ─────────────────────────────────────────
        # Light block BEFORE calling inference so user sees step 3 during the wait
        SHARED_STATE["ai_flow_step"] = 3
        yield _build_flow_html(3), gr.update(), "🧠 Model AI đang suy luận...", gr.update()

        # Actual inference (longest step — block stays lit for its real duration)
        display_img, log_text, log_data = start_inference(
            sys_device, sys_model_path, sys_output_dir,
            image_input, conf_thres, line_width, font_size
        )

        # ── STEP 4: Output Handling ─────────────────────────────────────────
        SHARED_STATE["ai_flow_step"] = 4

        # Auto-reset to idle after 3 s (daemon thread — does not block return)
        def _auto_reset():
            time.sleep(3.0)
            SHARED_STATE["ai_flow_step"] = 0
        threading.Thread(target=_auto_reset, daemon=True).start()

        yield _build_flow_html(4), display_img, log_text, log_data


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
        # ui_ai_flow is first output so each generator yield updates it immediately
        outputs=[ui_ai_flow, ui_result_img, ui_log_output, ui_log_table],
    )

    if app is not None:
        timer = gr.Timer(0.3)
        
        def sync_ui():
            step = SHARED_STATE["ai_flow_step"]
            flow_html = _build_flow_html(step)
            
            plc_str = "🟢 Đã kết nối" if SHARED_STATE["plc_connected"] else "🔴 Mất kết nối"
            m1020_str = "🟢 1" if SHARED_STATE["plc_vars"]["M1020"] else "⚫ 0"
            m169_str = "🟢 1" if SHARED_STATE["plc_vars"]["M169"] else "⚫ 0"
            m170_str = "🟢 1" if SHARED_STATE["plc_vars"]["M170"] else "⚫ 0"
            m171_str = "🔴 1" if SHARED_STATE["plc_vars"]["M171"] else "⚫ 0"

            updates = [
                gr.update(value=flow_html),   # ui_ai_flow
                gr.update(),                   # ui_csi_image
                gr.update(),                   # ui_csi_status
                gr.update(),                   # ui_result_img
                gr.update(),                   # ui_log_output
                gr.update(),                   # ui_log_table
                gr.update(value=plc_str),      # ui_plc_status
                gr.update(value=m1020_str),    # ui_m1020
                gr.update(value=m169_str),     # ui_m169
                gr.update(value=m170_str),     # ui_m170
                gr.update(value=m171_str),     # ui_m171
            ]

            if SHARED_STATE["has_new_capture"]:
                SHARED_STATE["has_new_capture"] = False
                updates[1] = gr.update(value=SHARED_STATE["capture_path"])
                updates[2] = gr.update(value="✓ Đã chụp từ PLC")
            if SHARED_STATE["has_new_scan"]:
                SHARED_STATE["has_new_scan"] = False
                updates[3] = gr.update(value=SHARED_STATE["scan_img"])
                updates[4] = gr.update(value=SHARED_STATE["scan_log"])
                updates[5] = gr.update(value=SHARED_STATE["scan_data"])

            return tuple(updates)
            
        timer.tick(
            fn=sync_ui,
            inputs=[],
            outputs=[ui_ai_flow, ui_csi_image, ui_csi_status,
                     ui_result_img, ui_log_output, ui_log_table,
                     ui_plc_status, ui_m1020, ui_m169, ui_m170, ui_m171]
        )