import cv2
import threading
import time
import os
from core.dataset_manager import load_system_config

class FastCamera:
    """Quản lý camera CSI Jetson Nano với chất lượng 8MP"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.width = 3280
        self.height = 2464
        self.save_quality = 100
        
        self.frame = None
        self.is_running = False
        self.cap = None
        self.thread = None
        self.camera_available = False
        
    def start(self):
        """Khởi động camera"""
        if self.cap is not None:
            return self.camera_available  # Đã cố gắng khởi động rồi
        
        try:
            # Cố gắng dùng GStreamer pipeline tối ưu cho Jetson
            gst_str = (
                "nvarguscamerasrc sensor-id=0 "
                "ee-mode=2 ee-strength=1.0 "
                "tnr-mode=2 tnr-strength=1.0 "
                "! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=21/1 "
                "! nvvidconv flip-method=0 "
                "! video/x-raw, format=BGRx "
                "! videoconvert "
                "! video/x-raw, format=BGR ! appsink drop=True"
            )
            
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print("[Camera] Lỗi: Không thể mở GStreamer pipeline")
                self.cap = None
                return False
            
            self.is_running = True
            self.camera_available = True
            
            # Chạy luồng đọc ảnh liên tục (như testcam.py)
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print("[Camera] ✓ Camera CSI khởi động thành công")
            time.sleep(2)  # Chờ ISP ổn định (như testcam.py)
            return True
            
        except Exception as e:
            print(f"[Camera] Lỗi khởi động: {e}")
            self.camera_available = False
            return False
    
    def _update(self):
        """Cập nhật frame liên tục"""
        fail_count = 0
        while self.is_running:
            try:
                ret, img = self.cap.read()
                if ret:
                    self.frame = img
                    self.camera_available = True
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count > 150:  # ~1.5 giây liên tục không nhận được frame
                        if self.camera_available:
                            print("\n[Camera] ❌ Mất tín hiệu từ camera (nvargus-daemon treo). Hãy chạy: sudo systemctl restart nvargus-daemon")
                            self.camera_available = False
                    time.sleep(0.01)
            except Exception as e:
                print(f"[Camera] Lỗi đọc frame: {e}")
                time.sleep(0.01)
    
    def capture(self, output_dir="outputs"):
        """Chụp ảnh từ camera và lưu vào output_dir"""
        if not self.camera_available or self.frame is None:
            return None, "❌ Camera CSI không khả dụng"
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            snap = self.frame.copy()
            
            # Đọc cấu hình kích thước và chất lượng được chỉnh trên Web
            cfg = load_system_config()
            cam_w = int(cfg.get("cam_width", 3280))
            cam_h = int(cfg.get("cam_height", 2464))
            cam_quality = int(cfg.get("cam_quality", 95))
            
            # Thay đổi tỷ lệ/kích thước trước khi lưu (nếu cần thiết)
            if snap.shape[1] != cam_w or snap.shape[0] != cam_h:
                snap = cv2.resize(snap, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                
            filename = f"pcb_{int(time.time())}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, snap, [cv2.IMWRITE_JPEG_QUALITY, cam_quality])
            
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            msg = (
                f"✓ Chụp ảnh thành công\n"
                f"File: {filename}\n"
                f"Độ phân giải: {snap.shape[1]}x{snap.shape[0]}\n"
                f"Dung lượng: {size_mb:.2f} MB"
            )
            return filepath, msg
            
        except Exception as e:
            return None, f"❌ Lỗi chụp ảnh: {e}"
    
    def get_current_frame(self):
        """Lấy frame hiện tại"""
        if self.frame is not None:
            return self.frame.copy()
        return None
    
    def is_camera_available(self):
        """Kiểm tra xem camera có sẵn sàng không"""
        return self.camera_available
    
    def stop(self):
        """Dừng camera"""
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.camera_available = False


def get_camera():
    """Lấy instance camera (Singleton)"""
    return FastCamera()
