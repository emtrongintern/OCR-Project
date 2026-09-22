import os
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_vietnamese_document_parsing(image_path: str):
    # Kiểm tra đường dẫn và hỗ trợ tìm ảnh theo thư mục script
    if not os.path.exists(image_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(image_path))
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            print(f"Không tìm thấy ảnh tại: {image_path}")
            return

    # Đọc ảnh an toàn với đường dẫn tiếng Việt / Unicode trên Windows
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Không thể đọc file ảnh tại: {image_path}")
        return

    # 1. Khởi tạo mô hình OCR Tiếng Việt
    print("Đang tải mô hình OCR Tiếng Việt...")
    ocr_vi = PaddleOCR(use_textline_orientation=True, lang="vi")

    print("Đang khởi tạo Layout Engine...")
    try:
        from paddleocr import LayoutDetection
        layout_engine = LayoutDetection()
        is_paddlex_layout = True
    except ImportError:
        from paddleocr import PPStructure
        layout_engine = PPStructure(show_log=False)
        is_paddlex_layout = False

    print(f"Đang phân tích cấu trúc tài liệu: {image_path}...")
    h, w = img.shape[:2]
    regions = []

    if is_paddlex_layout:
        layout_results = list(layout_engine.predict(image_path))
        for res_item in layout_results:
            for b in res_item.get("boxes", []):
                bbox = b.get("coordinate", b.get("bbox", []))
                label = b.get("label", "text")
                score = b.get("score", 1.0)
                regions.append({"type": label, "bbox": bbox, "score": score})
    else:
        raw_res = layout_engine(img)
        for r in raw_res:
            regions.append({"type": r.get("type", "text"), "bbox": r.get("bbox", [])})

    # Nếu không tìm thấy vùng bố cục riêng (ví dụ ảnh cắt hẹp), chạy OCR toàn ảnh
    if not regions:
        print("Không phát hiện vùng bố cục phức tạp, tiến hành nhận diện OCR trực tiếp trên ảnh...")
        ocr_results = list(ocr_vi.predict(image_path))
        for r in ocr_results:
            texts = r.get("rec_texts", [])
            scores = r.get("rec_scores", [])
            for text, score in zip(texts, scores):
                print(f" ➜ {text} (score: {float(score):.2f})")
        return

    # 2. Quá trình ghép nối: Cắt ảnh theo Layout -> Đưa vào OCR Tiếng Việt
    for region in regions:
        layout_type = region.get("type", "unknown")
        bbox = region.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        print(f"\n--- [Vùng: {layout_type.upper()}] (bbox: [{x1}, {y1}, {x2}, {y2}]) ---")

        if layout_type.lower() in ["title", "text", "paragraph", "header", "footer"]:
            cropped_img = img[y1:y2, x1:x2]
            ocr_results = list(ocr_vi.predict(cropped_img))

            for r in ocr_results:
                texts = r.get("rec_texts", [])
                scores = r.get("rec_scores", [])
                for text, score in zip(texts, scores):
                    if float(score) > 0.4:
                        print(f" ➜ {text} (score: {float(score):.2f})")

        elif layout_type.lower() == "table":
            print(" ➜ [Bảng biểu]")

        elif layout_type.lower() == "figure":
            print(" ➜ [Hình ảnh]")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_image = os.path.join(base_dir, "test_ngang.jpg")

    if not os.path.exists(sample_image):
        sample_image = "OCR_Project/test_ngang.jpg"

    run_vietnamese_document_parsing(sample_image)