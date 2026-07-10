import gradio as gr
import json
import time
import os
import cv2
import tempfile
from core.dataset_manager import save_annotation, split_dataset, load_system_config, auto_slice_image, save_raw_image_to_log
from core.camera_service import get_camera

GALLERY_CSS = """
<style>
#box-transfer { display: none !important; }
#horizontal-gallery { 
    height: 220px !important; 
    overflow-x: auto !important;
    overflow-y: hidden !important; 
    border: 1px solid #444;
}
#horizontal-gallery [data-testid="grid"] { 
    display: flex !important; 
    flex-direction: row !important; 
    flex-wrap: nowrap !important; 
    width: max-content !important; 
    min-width: 100% !important;
    height: 100% !important;
    gap: 15px !important;
    padding: 10px !important;
}
#horizontal-gallery button, #horizontal-gallery [data-testid="grid"] > * { 
    flex: 0 0 180px !important; 
    width: 180px !important; 
    height: 190px !important;
}
#horizontal-gallery img { 
    height: 145px !important;
    object-fit: cover !important; 
    border-radius: 8px; 
}
#horizontal-gallery::-webkit-scrollbar { height: 10px; }
#horizontal-gallery::-webkit-scrollbar-thumb { background: #666; border-radius: 5px; }
</style>
"""

JS_ANNOTATOR = """
function() {
    const container = document.querySelector('#workspace-img');
    const img = container ? container.querySelector('img') : null;
    if (!img) return;

    let canvas = document.querySelector('#label-canvas');
    
    const ratio = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    const actualWidth = img.naturalWidth * ratio;
    const actualHeight = img.naturalHeight * ratio;
    const padX = (img.clientWidth - actualWidth) / 2;
    const padY = (img.clientHeight - actualHeight) / 2;

    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'label-canvas';
        canvas.style.position = 'absolute';
        canvas.style.cursor = 'crosshair';
        canvas.style.zIndex = '1000';
        img.parentElement.style.position = 'relative';
        img.parentElement.appendChild(canvas);
    }
    
    canvas.style.left = padX + 'px';
    canvas.style.top = padY + 'px';
    canvas.style.width = actualWidth + 'px';
    canvas.style.height = actualHeight + 'px';
    canvas.width = actualWidth;
    canvas.height = actualHeight;

    const ctx = canvas.getContext('2d');
    window.pcb_boxes = window.pcb_boxes || [];

    function drawAll() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        window.pcb_boxes.forEach(b => {
            ctx.strokeStyle = '#00ff00'; 
            ctx.lineWidth = 2;
            ctx.strokeRect(b[0]*canvas.width, b[1]*canvas.height, (b[2]-b[0])*canvas.width, (b[3]-b[1])*canvas.height);
        });
    }

    drawAll();

    canvas.onmousedown = (e) => { this.startX = e.offsetX; this.startY = e.offsetY; this.isDrawing = true; };
    canvas.onmousemove = (e) => {
        if (!this.isDrawing) return;
        drawAll(); 
        ctx.strokeStyle = '#ff0000'; 
        ctx.strokeRect(this.startX, this.startY, e.offsetX - this.startX, e.offsetY - this.startY);
    };

    canvas.onmouseup = (e) => {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        const normBox = [this.startX/canvas.width, this.startY/canvas.height, e.offsetX/canvas.width, e.offsetY/canvas.height];
        window.pcb_boxes.push(normBox);
        drawAll();
        
        const origBox = [normBox[0]*img.naturalWidth, normBox[1]*img.naturalHeight, normBox[2]*img.naturalWidth, normBox[3]*img.naturalHeight];
        const transfer = document.querySelector('#box-transfer textarea');
        if (transfer) {
            transfer.value = JSON.stringify(origBox);
            transfer.dispatchEvent(new Event('input', { bubbles: true }));
        }
    };
}
"""

def render(sys_dataset_path, sys_device, camera_available=False):
    gr.HTML(GALLERY_CSS) 
    
    image_buffer = gr.State([])
    annotated_images = gr.State([])
    current_image_path = gr.State("")
    current_anno_list = gr.State([])
    class_counts = gr.State({})
    used_classes_state = gr.State([]) 

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Label Statistics")
            ui_anno_display = gr.HTML("<i>No boxes drawn yet</i>")
            gr.Markdown("---")
            ui_train = gr.Slider(0, 100, 70, label="Train %")
            ui_val = gr.Slider(0, 100, 20, label="Val %")
            ui_test = gr.Slider(0, 100, 10, label="Test %")
            ui_bg_ratio = gr.Slider(0, 100, 10, label="Keep Empty/Background Images (%)")
            btn_split = gr.Button("EXECUTE SPLIT", variant="secondary")

        with gr.Column(scale=4):
            ui_gallery = gr.Gallery(label="Temporary Storage", elem_id="horizontal-gallery", columns=10, rows=1, height=220)
            with gr.Row():
                with gr.Column():
                    ui_file_input = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], height=150)
                with gr.Column():
                    ui_cam_input = gr.Image(label="Capture from Webcam", sources=["webcam"], type="numpy", height=100)
                    btn_csi_capture = gr.Button("Capture CSI Camera (Add to Temporary Storage)", variant="primary", interactive=camera_available)
                    
                    gr.Markdown("---")
                    gr.Markdown("### Quick Log Saving (No Labeling Needed)")
                    ui_log_status = gr.Radio(["OK", "NG"], label="PCB Status", value="OK")
                    with gr.Row():
                        btn_log_csi_direct = gr.Button("Capture CSI & Save Log Instantly", variant="primary", interactive=camera_available)
                        btn_log_workspace = gr.Button("Save Current Workspace Image to Log", variant="secondary")

            gr.Markdown("---")
            with gr.Row():
                ui_work_img = gr.Image(label="Workspace", elem_id="workspace-img", interactive=False, type="numpy")
                with gr.Column(visible=False) as ui_label_column:
                    ui_current_box_data = gr.Textbox(value="", elem_id="box-transfer", interactive=True)
                    ui_class_input = gr.Textbox(label="Enter Class (Press Enter)", placeholder="Example: short_circuit")
                    btn_confirm_box = gr.Button("CONFIRM BOX", variant="primary")
                    btn_undo_box = gr.Button("Undo Last Drawn Box", variant="secondary")

            with gr.Row():
                btn_save_final = gr.Button("SAVE ALL LABELS", variant="primary")
                ui_status = gr.Textbox(label="Status", interactive=False, value="Ready")

    def update_gallery_view(buffer, ann_list):
        return [(p, "✅") if p in ann_list else p for p in buffer]

    def add_from_files(files, current_buffer, ann_list):
        if not files: return update_gallery_view(current_buffer, ann_list), current_buffer, gr.update()
        new_buffer = list(current_buffer)
        for f in files:
            sliced_paths = auto_slice_image(f.name)
            for sp in sliced_paths:
                if sp not in new_buffer:
                    new_buffer.append(sp)
        return update_gallery_view(new_buffer, ann_list), new_buffer, None

    ui_file_input.change(add_from_files, [ui_file_input, image_buffer, annotated_images], [ui_gallery, image_buffer, ui_file_input])

    def capture_csi(current_buffer, ann_list):
        cam = get_camera()
        if not cam.is_running:
            cam.start()
        tmp_dir = tempfile.gettempdir()
        filepath, msg = cam.capture(tmp_dir)
        if filepath and os.path.exists(filepath):
            new_buffer = []  # Xoá bộ nhớ tạm cũ, chỉ hiển thị ảnh vừa chụp
            sliced_paths = auto_slice_image(filepath)
            for sp in sliced_paths:
                if sp not in new_buffer:
                    new_buffer.append(sp)
            return update_gallery_view(new_buffer, ann_list), new_buffer
        return update_gallery_view(current_buffer, ann_list), current_buffer

    btn_csi_capture.click(capture_csi, [image_buffer, annotated_images], [ui_gallery, image_buffer])

    def add_from_webcam(img_array, current_buffer, ann_list):
        if img_array is None: return update_gallery_view(current_buffer, ann_list), current_buffer, None
        tmp_dir = tempfile.gettempdir()
        filepath = os.path.join(tmp_dir, f"webcam_{int(time.time())}.jpg")
        cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        new_buffer = list(current_buffer)
        sliced_paths = auto_slice_image(filepath, enable_sahi)
        for sp in sliced_paths:
            if sp not in new_buffer:
                new_buffer.append(sp)
        return update_gallery_view(new_buffer, ann_list), new_buffer, None

    ui_cam_input.change(add_from_webcam, [ui_cam_input, image_buffer, annotated_images], [ui_gallery, image_buffer, ui_cam_input])

    def on_gallery_select(evt: gr.SelectData, buffer):
        img_path = buffer[evt.index] if evt.index < len(buffer) else ""
        return img_path, img_path, gr.update(visible=True), [], {}, "<i>Ready to label</i>"

    JS_CLEAR_BOXES = "function(){ window.pcb_boxes = []; const canvas = document.querySelector('#label-canvas'); if(canvas){ canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height); } }"
    ui_gallery.select(on_gallery_select, [image_buffer], [current_image_path, ui_work_img, ui_label_column, current_anno_list, class_counts, ui_anno_display]).then(None, None, None, js=JS_CLEAR_BOXES)
    ui_work_img.change(fn=None, js=JS_ANNOTATOR)

    def confirm_one_box(box_json, label_name, anno_list, counts):
        if not box_json or not label_name.strip():
            return label_name, anno_list, counts, "<i>Must draw a box and enter label</i>"
        
        try:
            box = json.loads(box_json)
        except Exception:
            return label_name, anno_list, counts, "<i>Error reading box coordinates</i>"
        if not isinstance(box, list) or len(box) != 4:
            return label_name, anno_list, counts, "<i>Invalid box</i>"
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
            return label_name, anno_list, counts, "<i>Box too small, please redraw</i>"

        new_anno_list = list(anno_list)
        clean_label = label_name.strip()
        new_anno_list.append({"box": [x1, y1, x2, y2], "label": clean_label})
        
        new_counts = dict(counts)
        new_counts[clean_label] = new_counts.get(clean_label, 0) + 1
        
        # Trả về định dạng danh sách <ul><li> như code cũ của bạn
        html = "<ul>" + "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in new_counts.items()]) + "</ul>"
        return "", new_anno_list, new_counts, html

    ui_class_input.submit(confirm_one_box, [ui_current_box_data, ui_class_input, current_anno_list, class_counts], [ui_class_input, current_anno_list, class_counts, ui_anno_display])
    btn_confirm_box.click(confirm_one_box, [ui_current_box_data, ui_class_input, current_anno_list, class_counts], [ui_class_input, current_anno_list, class_counts, ui_anno_display])

    def undo_last_box(anno_list, counts):
        if not anno_list:
            return anno_list, counts, "<i>No boxes drawn yet</i>"
        last_ann = anno_list.pop()
        label = last_ann['label']
        if label in counts:
            counts[label] -= 1
            if counts[label] <= 0:
                del counts[label]
        html = "<ul>" + "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in counts.items()]) + "</ul>" if counts else "<i>No boxes drawn yet</i>"
        return anno_list, counts, html

    JS_UNDO = "function(){ if(window.pcb_boxes && window.pcb_boxes.length > 0) { window.pcb_boxes.pop(); } const canvas = document.querySelector('#label-canvas'); if(canvas) { const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); window.pcb_boxes.forEach(b => { ctx.strokeStyle='#00ff00'; ctx.lineWidth=2; ctx.strokeRect(b[0]*canvas.width, b[1]*canvas.height, (b[2]-b[0])*canvas.width, (b[3]-b[1])*canvas.height); }); } }"
    btn_undo_box.click(undo_last_box, [current_anno_list, class_counts], [current_anno_list, class_counts, ui_anno_display]).then(None, None, None, js=JS_UNDO)

    def handle_save_annotation(image, anno_list, dataset_path, used_classes, current_img_path, ann_list, current_buffer):
        status, new_used_classes = save_annotation(image, anno_list, dataset_path, used_classes)
        new_ann_list = list(ann_list)
        if "THÀNH CÔNG" in status and current_img_path and current_img_path not in new_ann_list:
            new_ann_list.append(current_img_path)
        gallery_items = update_gallery_view(current_buffer, new_ann_list)
        return status, new_used_classes, new_ann_list, gallery_items

    btn_save_final.click(
        fn=handle_save_annotation, 
        inputs=[ui_work_img, current_anno_list, sys_dataset_path, used_classes_state, current_image_path, annotated_images, image_buffer], 
        outputs=[ui_status, used_classes_state, annotated_images, ui_gallery]
    )
    btn_split.click(fn=split_dataset, inputs=[sys_dataset_path, ui_train, ui_val, ui_test, ui_bg_ratio], outputs=[ui_status])
    def handle_log_workspace(image, status):
        if image is None:
            return "ERROR: No image on Workspace screen."
        return save_raw_image_to_log(image, status_folder=status)

    btn_log_workspace.click(
        fn=handle_log_workspace,
        inputs=[ui_work_img, ui_log_status],
        outputs=[ui_status]
    )

    # 2. Chụp từ Camera CSI và ném thẳng vào ổ Log
    def handle_log_csi_direct(status):
        cam = get_camera()
        if not cam.is_running:
            cam.start()
            
        tmp_dir = tempfile.gettempdir()
        filepath, msg = cam.capture(tmp_dir)
        
        if filepath and os.path.exists(filepath):
            # Đọc ảnh bằng OpenCV
            img = cv2.imread(filepath)
            if img is not None:
                # Chuyển đổi BGR (OpenCV) sang RGB (chuẩn chung của app) để hàm log tự xử lý
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return save_raw_image_to_log(img_rgb, status_folder=status)
            return "ERROR: Cannot read captured image."
        return f"Camera ERROR: {msg}"

    btn_log_csi_direct.click(
        fn=handle_log_csi_direct,
        inputs=[ui_log_status],
        outputs=[ui_status]
    )