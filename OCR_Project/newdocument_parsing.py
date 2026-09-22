import os
import sys
import argparse
import json
import re
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Cấu hình encoding UTF-8 cho Windows Terminal để không bị lỗi font tiếng Việt
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass



# 1. BẢNG ĐỊNH TUYẾN TRƯỜNG DỮ LIỆU (FIELD ROUTING CONFIGURATION)

# Nhóm các trường chữ số, mã định danh, ngày tháng -> Giao cho PP-OCRv6
PPOCR_FIELD_KEYWORDS = {
    'id', 'id_number', 'cccd', 'cmnd', 'code', 'number', 'so_dinh_danh',
    'ma_so', 'phone', 'so_dien_thoai', 'dob', 'ngay_sinh', 'date', 'ngay_thang',
    'stt', 'serial', 'so_ho_chieu', 'passport_no', 'tax_code', 'ma_so_thue'
}

# Nhóm các trường họ tên, văn bản tiếng Việt -> Giao cho VietOCR
VIETOCR_FIELD_KEYWORDS = {
    'name', 'ho_ten', 'full_name', 'ten', 'ho_va_ten',
    'address', 'dia_chi', 'que_quan', 'noi_sinh', 'noi_tru', 'thuong_tru',
    'gender', 'gioi_tinh', 'nationality', 'quoc_tich', 'dan_toc',
    'job', 'nghe_nghiep', 'title', 'tieu_de', 'text', 'ghi_chu'
}


def determine_ocr_engine(field_name: str, draft_text: str = "") -> str:
    """
    Quyết định chọn model OCR phù hợp dựa trên nhãn trường từ YOLO.
    Nếu nhãn chưa rõ, dùng thêm heuristic kiểm tra nội dung text nháp.
    
    Returns:
        'ppocr_v6' hoặc 'vietocr'
    """
    field_lower = field_name.lower().strip()
    
    # 1. Kiểm tra theo nhãn trường do YOLO phân loại
    for kw in PPOCR_FIELD_KEYWORDS:
        if kw in field_lower:
            return "ppocr_v6"

    for kw in VIETOCR_FIELD_KEYWORDS:
        if kw in field_lower:
            return "vietocr"

    # 2. Heuristic fallback nếu nhãn lạ hoặc không xác định (ví dụ nhãn 'field_1', 'box')
    if draft_text:
        text_clean = draft_text.replace(" ", "")
        # Nếu chuỗi thuần số từ 6 ký tự trở lên hoặc dạng mã A-Z0-9
        if re.fullmatch(r'[0-9\.\-\/]+', text_clean) or (re.fullmatch(r'[A-Z0-9\-]+', text_clean) and len(text_clean) >= 4):
            return "ppocr_v6"

    # Mặc định văn bản tiếng Việt chuyển sang VietOCR
    return "vietocr"



# 2. MÔ HÌNH PHÁT HIỆN BỐ CỤC (YOLO LAYOUT DETECTOR)

class YOLOLayoutDetector:
    """
    Module phát hiện bố cục các trường dữ liệu bằng mô hình YOLO fine-tuned.
    Xử lý tốt tình huống tài liệu in hơi lệch, trôi lề, nghiêng nhẹ.
    """
    def __init__(self, model_path: str = None, conf_thresh: float = 0.4, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.is_simulated = False
        
        # Tìm kiếm đường dẫn file weights phù hợp
        candidate_paths = []
        if model_path:
            candidate_paths.append(model_path)
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths.extend([
            os.path.join(base_dir, "yolo26.pt"),
            os.path.join(base_dir, "best.pt"),
            os.path.join(base_dir, "yolo_layout.pt"),
            os.path.join(base_dir, "weights", "best.pt"),
            "yolo26.pt",
            "best.pt"
        ])
        
        chosen_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                chosen_path = path
                break
                
        if chosen_path:
            try:
                from ultralytics import YOLO
                print(f"[YOLO] Đang nạp weights YOLO Layout Fine-tuned từ: {chosen_path}")
                self.model = YOLO(chosen_path)
                print("[YOLO] Đã nạp thành công mô hình YOLO Layout!")
            except Exception as e:
                print(f"[YOLO Warning] Không thể nạp weights YOLO ({e}). Sẽ sử dụng Fallback Layout Engine.")
                self.is_simulated = True
        else:
            print("[YOLO Warning] Chưa phát hiện file weights YOLO fine-tune ('yolo26.pt' hoặc 'best.pt').")
            print("              Hệ thống sẽ chạy ở chế độ Fallback Layout (PaddleOCR Det / Mock Field) để kiểm thử.")
            self.is_simulated = True

    def detect(self, img_bgr: np.ndarray):
        """
        Dự đoán các vùng trường dữ liệu trên ảnh.
        
        Returns:
            list of dict: [
                {
                    "field_name": "id" / "ho_ten" / ...,
                    "box": [x1, y1, x2, y2],
                    "confidence": float
                }, ...
            ]
        """
        h, w = img_bgr.shape[:2]
        detected_fields = []

        if self.model and not self.is_simulated:
            results = self.model.predict(
                source=img_bgr,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    field_name = r.names.get(cls_id, f"field_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]
                    
                    detected_fields.append({
                        "field_name": field_name,
                        "box": [x1, y1, x2, y2],
                        "confidence": conf
                    })
        else:
            # Chế độ mô phỏng / Fallback khi chưa có file weights .pt
            detected_fields = self._fallback_detection(img_bgr)

        return detected_fields

    def _fallback_detection(self, img_bgr: np.ndarray):
        """
        Fallback layout detector sử dụng hình ảnh mẫu hiện tại để minh họa quy trình.
        """
        h, w = img_bgr.shape[:2]
        fields = []
        
        # Mẫu 1: Nhận diện trường ID (nửa trên bên phải hoặc vị trí tiêu biểu)
        # Giả lập 2 trường layout điển hình để test logic routing: ID và Họ tên
        if h > 200 and w > 200:
            # Trường Họ tên
            fields.append({
                "field_name": "ho_ten",
                "box": [int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.45)],
                "confidence": 0.92
            })
            # Trường Số / Mã định danh ID
            fields.append({
                "field_name": "id",
                "box": [int(w * 0.1), int(h * 0.55), int(w * 0.9), int(h * 0.85)],
                "confidence": 0.95
            })
        else:
            fields.append({
                "field_name": "text",
                "box": [0, 0, w, h],
                "confidence": 0.90
            })
        return fields



# 3. TIỆN ÍCH CẮT VÙNG ẢNH CÓ PADDING (GIẢI PHÁP CHO IN HƠI LỆCH)

def crop_field_with_padding(img_bgr: np.ndarray, box: list, padding_x: int = 6, padding_y: int = 4):
    """
    Cắt vùng ảnh theo bounding box của YOLO kết hợp thêm lề (padding) an toàn.
    Lợi ích:
    - Khi in lệch nhẹ hoặc YOLO dự đoán box sát mép chữ, padding giúp không bị
      mất nét đầu/cuối của tên hoặc số ID.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Mở rộng vùng cắt theo padding
    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(w, x2 + padding_x)
    y2_pad = min(h, y2 + padding_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, [x1, y1, x2, y2]

    cropped = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    return cropped, [x1_pad, y1_pad, x2_pad, y2_pad]



# 4. HỆ THỐNG OCR KẾT HỢP (PP-OCRV6 + VIETOCR)

class HybridDocumentOCR:
    """
    Quản lý các mô hình OCR chuyên trách:
    - PP-OCRv6: Chuyên số, ID, mã số, ngày sinh
    - VietOCR: Chuyên họ tên, văn bản tiếng Việt có dấu
    """
    def __init__(self, vietocr_weights_path: str = None):
        print("=" * 70)
        print("KHỞI TẠO CÁC MÔ HÌNH OCR CHUYÊN TRÁCH...")
        print("=" * 70)

        # 1. Khởi tạo PP-OCRv6 (Tối ưu tốc độ, tắt xoay toàn trang vì ảnh đã crop)
        print("[1/2] Đang nạp mô hình PP-OCRv6 (PaddleOCR)...")
        from paddleocr import PaddleOCR
        self.ppocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='en'  # Model English đọc số và ký tự ID cực sạch, không lỗi dấu
        )
        print("[1/2] Đã nạp thành công PP-OCRv6!")

        # 2. Khởi tạo VietOCR (VGG-Transformer)
        print("[2/2] Đang nạp mô hình VietOCR (VGG-Transformer)...")
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        config = Cfg.load_config_from_name('vgg_transformer')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if vietocr_weights_path and os.path.exists(vietocr_weights_path):
            config['weights'] = vietocr_weights_path
        else:
            default_weights = os.path.join(base_dir, 'vgg_transformer.pth')
            if os.path.exists(default_weights):
                config['weights'] = default_weights
            else:
                print(f"[VietOCR Info] Sử dụng weights mặc định tải từ internet.")
                
        config['device'] = 'cpu'
        self.vietocr_engine = Predictor(config)
        print("[2/2] Đã nạp thành công VietOCR!")
        print("=" * 70)

    def recognize_with_ppocr(self, cropped_bgr: np.ndarray) -> tuple:
        """
        Nhận diện bằng PP-OCRv6. Thích hợp cho số, ID, mã số.
        """
        try:
            results = list(self.ppocr_engine.predict(cropped_bgr))
            texts = []
            scores = []
            if results and isinstance(results, list):
                for res in results:
                    if isinstance(res, dict):
                        t_list = res.get('rec_texts', [])
                        s_list = res.get('rec_scores', [])
                        texts.extend(t_list)
                        scores.extend(s_list)
            
            final_text = " ".join(texts).strip()
            avg_score = float(np.mean(scores)) if scores else 0.95
            return final_text, avg_score
        except Exception as e:
            print(f"[PP-OCRv6 Error]: {e}")
            return "", 0.0

    def recognize_with_vietocr(self, cropped_bgr: np.ndarray) -> tuple:
        """
        Nhận diện bằng VietOCR. Thích hợp cho Họ tên, địa chỉ tiếng Việt có dấu.
        """
        try:
            # VietOCR nhận ảnh định dạng PIL RGB
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cropped_rgb)
            text = self.vietocr_engine.predict(pil_img)
            return text.strip(), 0.98
        except Exception as e:
            print(f"[VietOCR Error]: {e}")
            return "", 0.0



# 5. PIPELINE XỬ LÝ TOÀN DIỆN (YOLO DETECT -> ROUTING -> OCR -> EXPORT)
def run_pipeline(image_path: str, yolo_model_path: str = None, output_dir: str = "output"):
    """
    Thực thi toàn bộ quy trình nhận dạng tài liệu theo layout.
    """
    if not os.path.exists(image_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(image_path))
        if os.path.exists(alt):
            image_path = alt
        else:
            raise FileNotFoundError(f"Không tìm thấy file ảnh tài liệu tại: {image_path}")

    # 1. Đọc ảnh an toàn với đường dẫn tiếng Việt trên Windows
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Không thể giải mã file ảnh: {image_path}")

    h_img, w_img = img_bgr.shape[:2]
    print(f"\n[BẮT ĐẦU XỬ LÝ] Tài liệu: {image_path} (Kích thước: {w_img}x{h_img}px)")

    # 2. Phát hiện layout bằng YOLO Fine-tuned
    layout_detector = YOLOLayoutDetector(model_path=yolo_model_path)
    detected_fields = layout_detector.detect(img_bgr)
    print(f"[YOLO] Phát hiện được {len(detected_fields)} trường layout trên tài liệu.\n")

    # 3. Khởi tạo engine OCR
    ocr_system = HybridDocumentOCR()

    results_data = []
    annotated_img = img_bgr.copy()

    print("=" * 75)
    print(f"{'TRƯỜNG':<15} | {'MODEL CHỌN':<15} | {'ĐỘ TIN CẬY':<10} | {'KẾT QUẢ NHẬN DIỆN'}")
    print("=" * 75)

    # 4. Xử lý từng trường theo nguyên tắc Routing
    for i, field in enumerate(detected_fields):
        field_name = field["field_name"]
        raw_box = field["box"]
        det_conf = field["confidence"]

        # Cắt ảnh có padding an toàn chống lệch in
        cropped_patch, padded_box = crop_field_with_padding(img_bgr, raw_box, padding_x=8, padding_y=4)
        if cropped_patch is None or cropped_patch.size == 0:
            continue

        # Định tuyến chọn model phù hợp
        chosen_engine = determine_ocr_engine(field_name)

        if chosen_engine == "ppocr_v6":
            recognized_text, ocr_conf = ocr_system.recognize_with_ppocr(cropped_patch)
            model_display = "PP-OCRv6 (ID/Số)"
            box_color = (255, 120, 0)  # Xanh dương / Cyan
        else:
            recognized_text, ocr_conf = ocr_system.recognize_with_vietocr(cropped_patch)
            model_display = "VietOCR (Họ tên)"
            box_color = (0, 180, 0)    # Xanh lá cây

        print(f"{field_name:<15} | {model_display:<15} | {ocr_conf:<10.2f} | {recognized_text}")

        results_data.append({
            "order": i + 1,
            "field_name": field_name,
            "box": padded_box,
            "yolo_confidence": round(det_conf, 3),
            "ocr_model": model_display,
            "ocr_confidence": round(ocr_conf, 3),
            "text": recognized_text
        })

        # 5. Vẽ trực quan lên ảnh kết quả
        px1, py1, px2, py2 = padded_box
        cv2.rectangle(annotated_img, (px1, py1), (px2, py2), box_color, 2)
        
        # Nhãn hiển thị trên ảnh
        tag = f"{field_name} [{chosen_engine.upper()}]: {recognized_text}"
        # Nền nhãn
        cv2.rectangle(annotated_img, (px1, max(0, py1 - 22)), (min(w_img, px1 + len(tag) * 9), py1), box_color, -1)
        cv2.putText(
            annotated_img,
            tag,
            (px1 + 2, max(12, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    print("=" * 75)

    # 6. Lưu kết quả ra thư mục output
    os.makedirs(output_dir, exist_ok=True)
    stem_name = Path(image_path).stem

    # Lưu file JSON
    json_path = os.path.join(output_dir, f"{stem_name}_layout_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "document_path": image_path,
            "image_size": {"width": w_img, "height": h_img},
            "total_fields": len(results_data),
            "fields": results_data
        }, f, ensure_ascii=False, indent=2)

    # Lưu ảnh trực quan
    img_out_path = os.path.join(output_dir, f"{stem_name}_annotated.jpg")
    cv2.imencode('.jpg', annotated_img)[1].tofile(img_out_path)

    print(f"\n[XUẤT KẾT QUẢ THÀNH CÔNG]:")
    print(f" [+] File JSON trích xuất trường: {json_path}")
    print(f" [+] Ảnh vẽ khung và nhãn:        {img_out_path}")
    print("=" * 75)

    return results_data


# ============================================================================
# 6. ĐIỂM KHỞI CHẠY (CLI ENTRYPOINT)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Layout Pipeline: YOLO Fine-tuned + PP-OCRv6 + VietOCR")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh tài liệu")
    parser.add_argument("--yolo-model", type=str, default=None, help="Đường dẫn file weights YOLO (yolo26.pt / best.pt)")
    parser.add_argument("--output", type=str, default="output", help="Thư mục lưu kết quả")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nếu không truyền qua tham số dòng lệnh, tự tìm file ảnh có sẵn để test
    if args.image:
        target_image = args.image
    else:
        test_candidates = [
            os.path.join(base_dir, "testchuviettay.png"),
        ]
        target_image = next((c for c in test_candidates if os.path.exists(c)), "OCR_Project/test_document.png")

    run_pipeline(
        image_path=target_image,
        yolo_model_path=args.yolo_model,
        output_dir=args.output
    )