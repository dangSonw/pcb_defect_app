import gradio as gr
import json
import time
import os
from core.dataset_manager import save_annotation, split_dataset

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
    height: 160px !important;
}
#horizontal-gallery img { object-fit: cover !important; border-radius: 8px; }
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
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'label-canvas';
        canvas.style.position = 'absolute';
        canvas.style.top = '0px';
        canvas.style.left = '0px';
        canvas.style.cursor = 'crosshair';
        canvas.style.zIndex = '1000';
        img.parentElement.style.position = 'relative';
        img.parentElement.appendChild(canvas);
    }
    
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
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

def render(sys_dataset_path, sys_device):
    gr.HTML(GALLERY_CSS) 
    
    image_buffer = gr.State([])
    current_anno_list = gr.State([])
    class_counts = gr.State({})
    used_classes_state = gr.State([]) 

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Thống kê Nhãn")
            ui_anno_display = gr.HTML("<i>Chưa có box nào</i>")
            gr.Markdown("---")
            ui_train = gr.Slider(0, 100, 70, label="Train %")
            ui_val = gr.Slider(0, 100, 20, label="Val %")
            ui_test = gr.Slider(0, 100, 10, label="Test %")
            btn_split = gr.Button("THỰC THI CHIA", variant="secondary")

        with gr.Column(scale=4):
            ui_gallery = gr.Gallery(label="Bộ nhớ tạm", elem_id="horizontal-gallery", columns=10, rows=1, height=220)
            with gr.Row():
                ui_file_input = gr.File(label="Tải ảnh", file_count="multiple", file_types=["image"], height=150)
                ui_cam_input = gr.Image(label="Chụp Camera", sources=["webcam"], type="numpy")

            gr.Markdown("---")
            with gr.Row():
                ui_work_img = gr.Image(label="Workspace", elem_id="workspace-img", interactive=False, type="numpy")
                with gr.Column(visible=False) as ui_label_column:
                    ui_current_box_data = gr.Textbox(value="", elem_id="box-transfer", interactive=True)
                    ui_class_input = gr.Textbox(label="Nhập Class (Nhấn Enter)", placeholder="Ví dụ: short_circuit")
                    btn_confirm_box = gr.Button("XÁC NHẬN BOX", variant="primary")

            with gr.Row():
                btn_save_final = gr.Button("LƯU TOÀN BỘ GÁN NHÃN", variant="primary")
                ui_status = gr.Textbox(label="Trạng thái", interactive=False, value="Sẵn sàng")

    def add_from_files(files, current_buffer):
        if not files: return current_buffer, current_buffer, gr.update()
        new_buffer = list(current_buffer)
        existing_names = {os.path.basename(p) for p in new_buffer}
        for f in files:
            if os.path.basename(f.name) not in existing_names:
                new_buffer.append(f.name)
        # Reset ui_file_input về None để cho phép tải lại file cùng tên mà không bị kẹt
        return new_buffer, new_buffer, None

    ui_file_input.change(add_from_files, [ui_file_input, image_buffer], [ui_gallery, image_buffer, ui_file_input])

    def on_gallery_select(evt: gr.SelectData):
        img_path = evt.value['image']['path'] if isinstance(evt.value, dict) else evt.value
        return img_path, gr.update(visible=True), [], {}, "<i>Sẵn sàng gán nhãn</i>"

    ui_gallery.select(on_gallery_select, None, [ui_work_img, ui_label_column, current_anno_list, class_counts, ui_anno_display])
    ui_work_img.change(fn=None, js=JS_ANNOTATOR)

    def confirm_one_box(box_json, label_name, anno_list, counts):
        if not box_json or not label_name.strip():
            return label_name, anno_list, counts, "<i>Cần vẽ box và nhập nhãn</i>"
        
        try:
            box = json.loads(box_json)
        except Exception:
            return label_name, anno_list, counts, "<i>Loi doc toa do box</i>"
        if not isinstance(box, list) or len(box) != 4:
            return label_name, anno_list, counts, "<i>Box khong hop le</i>"
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
            return label_name, anno_list, counts, "<i>Box qua nho, vui long ve lai</i>"

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

    btn_save_final.click(fn=save_annotation, inputs=[ui_work_img, current_anno_list, sys_dataset_path, used_classes_state], outputs=[ui_status, used_classes_state])
    btn_split.click(fn=split_dataset, inputs=[sys_dataset_path, ui_train, ui_val, ui_test], outputs=[ui_status])