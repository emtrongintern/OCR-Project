"""
Pipeline OCR kết hợp PP-DocLayoutV3 + PP-StructureV3 với VietOCR + PP-OCRv6
=============================================================================
Quy trình thực hiện:
1. PP-DocLayoutV3 + PP-StructureV3:
   - Chia Khung (Layout Blocks): Phát hiện các vùng cấu trúc (tiêu đề, đoạn văn, bảng biểu, hình ảnh, con dấu...).
   - Chia Ô (Cells / Text boxes):
     + Ô trong bảng biểu (Table Cells) từ module Table Recognition (RT-DETR / SLANet).
     + Ô chữ / dòng chữ (Text Boxes) từ module Text Detection của PP-OCRv6.
     + Lọc nhiễu, khử trùng lặp (IoU Deduplication) và sắp xếp thứ tự đọc tự nhiên (Reading Order).

2. Nhận diện chữ kết hợp VietOCR + PP-OCRv6:
   - Cắt (crop) từng ô ảnh kèm padding an toàn chống lệch nét.
   - Nhận diện song song hoặc định tuyến thông minh:
     + VietOCR (VGG-Transformer): Chuyên nhận diện chữ tiếng Việt có dấu, họ tên, địa chỉ, văn bản.
     + PP-OCRv6 (PaddleOCR): Chuyên nhận diện chữ số, mã barcode/ID, ngày tháng, mã ký tự không dấu.
     + Hợp nhất (Arbitration/Fusion): Tự động chọn kết quả tối ưu nhất cho từng ô.

3. Xuất 1 file ảnh gồm 2 ảnh ghép ngang (Side-by-Side Dual-View Image):
   - Bên trái: Ảnh gốc với các ô đã được quét (vẽ bounding box màu sắc + đánh số thứ tự #1, #2, #3...).
   - Bên phải: Ảnh chữ viết ứng với từng ô đã quét (tọa độ tương ứng, render font tiếng Việt chuẩn Unicode,
     kèm nhãn model nhận diện VietOCR / PP-OCRv6).
=============================================================================
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Đảm bảo Terminal Windows xuất UTF-8 chuẩn xác
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================================
# 1. TIỆN ÍCH HÌNH HỌC VÀ LỌC BỘ NHIỄU (GEOMETRY & FILTERING)
# ============================================================================

def compute_iou(box_a, box_b):
    """Tính chỉ số IoU và tỷ lệ chồng lấn diện tích giữa 2 bounding box [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    if box_a_area <= 0 or box_b_area <= 0:
        return 0.0

    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)
    coverage = inter_area / float(min(box_a_area, box_b_area) + 1e-6)
    return max(iou, coverage)


def deduplicate_boxes(boxes_with_meta, iou_thresh=0.75):
    """
    Loại bỏ các bounding box bị trùng lặp hoặc lồng hoàn toàn vào nhau.
    Giữ lại box có diện tích hoặc độ tin cậy tốt hơn.
    """
    if not boxes_with_meta:
        return []

    # Sắp xếp theo diện tích giảm dần
    sorted_items = sorted(
        boxes_with_meta,
        key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
        reverse=True
    )

    kept = []
    for item in sorted_items:
        box = item["box"]
        is_duplicate = False
        for kept_item in kept:
            if compute_iou(box, kept_item["box"]) > iou_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(item)

    return kept


def sort_reading_order(boxes_with_meta, y_tol=20):
    """
    Sắp xếp các ô theo thứ tự đọc tự nhiên từ trên xuống dưới, từ trái sang phải.
    Nhóm các ô cùng dòng (theo độ chênh lệch trục Y) và sắp xếp theo X.
    """
    if not boxes_with_meta:
        return []

    # Sắp xếp theo Y trước
    sorted_y = sorted(boxes_with_meta, key=lambda item: (item["box"][1], item["box"][0]))
    lines = []

    for item in sorted_y:
        box = item["box"]
        y1, y2 = box[1], box[3]
        placed = False

        for line in lines:
            avg_y = sum(x["box"][1] for x in line) / len(line)
            avg_h = sum((x["box"][3] - x["box"][1]) for x in line) / len(line)
            tol = max(y_tol, avg_h * 0.45)
            if abs(y1 - avg_y) < tol:
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    ordered = []
    for line in lines:
        line_sorted = sorted(line, key=lambda item: item["box"][0])
        ordered.extend(line_sorted)

    # Gán lại ID thứ tự đọc tuần tự: 1, 2, 3...
    for idx, item in enumerate(ordered):
        item["order_id"] = idx + 1

    return ordered


def crop_image_patch(image, box, pad_x=6, pad_y=4):
    """Cắt vùng ảnh theo box với padding an toàn tránh mất nét rìa chữ."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, [x1, y1, x2, y2]

    patch = image[y1_pad:y2_pad, x1_pad:x2_pad]
    return patch, [x1_pad, y1_pad, x2_pad, y2_pad]


# ============================================================================
# 2. BỘ PHÂN ĐOẠN KHUNG VÀ Ô (PP-DOCLAYOUTV3 + PP-STRUCTUREV3)
# ============================================================================

class DocumentStructureSegmenter:
    """
    Sử dụng PP-StructureV3 kết hợp mô hình phân tích bố cục PP-DocLayoutV3
    để trích xuất:
    - Danh sách Khung (Layout Frames): Khối tiêu đề, đoạn văn bản, bảng biểu, ảnh,...
    - Danh sách Ô (Cells): Ô trong bảng (Table Cells) và Ô chữ (Text line cells).
    """
    def __init__(self):
        print("=" * 75)
        print("1. KHỞI TẠO PIPELINE PP-STRUCTUREV3 KẾT HỢP PP-DOCLAYOUTV3...")
        print("=" * 75)
        from paddleocr import PPStructureV3

        self.pipeline = PPStructureV3(
            layout_detection_model_name="PP-DocLayoutV3",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_table_recognition=True,
        )
        print("[+] Khởi tạo thành công PP-StructureV3 (Layout: PP-DocLayoutV3)!")

    def segment(self, image_path: str):
        """
        Thực hiện phân tích cấu trúc tài liệu.
        
        Returns:
            processed_bgr (np.ndarray): Ảnh đã căn chỉnh góc xoay.
            frames (list): Danh sách các khung bố cục lớn.
            cells (list): Danh sách các ô chi tiết cần nhận diện chữ.
        """
        print(f"\n[+] Đang chia khung và ô trên tài liệu: {image_path}...")
        results = list(self.pipeline.predict(input=image_path))
        if not results:
            raise RuntimeError(f"Không có kết quả trả về từ PP-StructureV3 cho file {image_path}")

        res = results[0]
        
        # Đọc ảnh gốc bằng OpenCV an toàn với ký tự Unicode
        orig_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if orig_img is None:
            raise ValueError(f"Không thể đọc file ảnh tại: {image_path}")

        # Kiểm tra góc xoay tự động từ Doc Preprocessor
        angle = 0
        if isinstance(res, dict):
            doc_prep = res.get("doc_preprocessor_res", {})
            if isinstance(doc_prep, dict):
                angle = doc_prep.get("angle", 0)
        elif hasattr(res, "get"):
            doc_prep = res.get("doc_preprocessor_res", {})
            if isinstance(doc_prep, dict):
                angle = doc_prep.get("angle", 0)

        # Căn chỉnh xoay ảnh nếu cần để tọa độ bounding box khớp 100%
        processed_img = orig_img.copy()
        if angle == 90:
            processed_img = cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            processed_img = cv2.rotate(processed_img, cv2.ROTATE_180)
        elif angle == 270:
            processed_img = cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_h, img_w = processed_img.shape[:2]

        # --------------------------------------------------------------------
        # 1. TRÍCH XUẤT CÁC KHUNG (LAYOUT FRAMES / BLOCKS) TỪ PP-DOCLAYOUTV3
        # --------------------------------------------------------------------
        frames = []
        layout_det_res = res.get("layout_det_res", {}) if hasattr(res, "get") else {}
        layout_boxes = layout_det_res.get("boxes", []) if isinstance(layout_det_res, dict) else []

        if not layout_boxes and hasattr(res, "get"):
            layout_boxes = res.get("parsing_res_list", [])

        for idx, block in enumerate(layout_boxes):
            if isinstance(block, dict):
                label = block.get("label", block.get("block_label", "text"))
                score = float(block.get("score", 1.0))
                coord = block.get("coordinate", block.get("block_bbox", []))
            else:
                label = getattr(block, "label", "text")
                score = float(getattr(block, "score", 1.0))
                coord = getattr(block, "bbox", [])

            if len(coord) >= 4:
                x1, y1, x2, y2 = [int(round(v)) for v in coord[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    frames.append({
                        "frame_id": idx + 1,
                        "label": label,
                        "score": round(score, 3),
                        "box": [x1, y1, x2, y2]
                    })

        # --------------------------------------------------------------------
        # 2. TRÍCH XUẤT CÁC Ô (CELLS / TEXT BOXES)
        # --------------------------------------------------------------------
        raw_cells = []

        # A. Trích xuất ô từ bảng biểu (Table Cells) nếu tài liệu có bảng
        table_res_list = res.get("table_res_list", []) if hasattr(res, "get") else []
        if table_res_list:
            for t_idx, table_item in enumerate(table_res_list):
                cell_box_list = table_item.get("cell_box_list", []) if isinstance(table_item, dict) else getattr(table_item, "cell_box_list", [])
                for c_idx, cbox in enumerate(cell_box_list):
                    if len(cbox) >= 4:
                        cx1, cy1, cx2, cy2 = [int(round(v)) for v in cbox[:4]]
                        cx1, cy1 = max(0, cx1), max(0, cy1)
                        cx2, cy2 = min(img_w, cx2), min(img_h, cy2)
                        cw, ch = cx2 - cx1, cy2 - cy1
                        if cw >= 12 and ch >= 10:
                            raw_cells.append({
                                "source": "table_cell",
                                "frame_label": f"table_{t_idx + 1}_cell",
                                "box": [cx1, cy1, cx2, cy2]
                            })

        # B. Trích xuất ô văn bản từ Overall OCR Res
        overall_ocr = res.get("overall_ocr_res", {}) if hasattr(res, "get") else {}
        if isinstance(overall_ocr, dict):
            rec_boxes = overall_ocr.get("rec_boxes", [])
            for rbox in rec_boxes:
                if len(rbox) >= 4:
                    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rbox[:4]]
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
                    rw, rh = rx2 - rx1, ry2 - ry1
                    if rw >= 14 and rh >= 10:
                        # Gắn nhãn layout frame cha cho ô chữ này
                        parent_label = "text"
                        for f in frames:
                            fx1, fy1, fx2, fy2 = f["box"]
                            # Nếu ô nằm phần lớn trong khung layout
                            if rx1 >= fx1 - 10 and ry1 >= fy1 - 10 and rx2 <= fx2 + 10 and ry2 <= fy2 + 10:
                                parent_label = f["label"]
                                break

                        raw_cells.append({
                            "source": "text_cell",
                            "frame_label": parent_label,
                            "box": [rx1, ry1, rx2, ry2]
                        })

        # Khử trùng lặp và sắp xếp thứ tự đọc tự nhiên
        deduped = deduplicate_boxes(raw_cells, iou_thresh=0.70)
        ordered_cells = sort_reading_order(deduped, y_tol=22)

        print(f"[+] PP-DocLayoutV3 chia được: {len(frames)} khung bố cục lớn.")
        print(f"[+] PP-StructureV3 chia được: {len(ordered_cells)} ô cần nhận diện chữ.")

        return processed_img, frames, ordered_cells


# ============================================================================
# 3. HỆ THỐNG NHẬN DIỆN CHỮ KẾT HỢP (VIETOCR + PP-OCRV6)
# ============================================================================

class DualOCRRecognizer:
    """
    Hệ thống nhận diện chữ kết hợp song song:
    - VietOCR: Transformer cực mạnh về tiếng Việt có dấu, họ tên, ngữ pháp tiếng Việt.
    - PP-OCRv6: Cực nhanh và chuẩn xác về số, mã định danh, ký tự ID, ngày tháng.
    """
    def __init__(self, vietocr_weights_path: str = None):
        print("\n" + "=" * 75)
        print("2. KHỞI TẠO CÁC MODEL NHẬN DIỆN CHỮ: VIETOCR VÀ PP-OCRV6...")
        print("=" * 75)

        # 1. VietOCR (VGG Transformer)
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        config = Cfg.load_config_from_name("vgg_transformer")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_weights = [
            vietocr_weights_path,
            os.path.join(base_dir, "vgg_transformer.pth"),
            os.path.join(os.getcwd(), "OCR_Project", "vgg_transformer.pth"),
            "vgg_transformer.pth"
        ]
        chosen_weights = next((p for p in candidate_weights if p and os.path.exists(p)), None)
        if chosen_weights:
            config["weights"] = chosen_weights
            print(f"[VietOCR] Đã tìm thấy weights: {chosen_weights}")
        else:
            print("[VietOCR] Tự động nạp weights mặc định.")

        config["device"] = "cpu"
        self.vietocr_predictor = Predictor(config)
        print("[+] Khởi tạo VietOCR (VGG-Transformer) thành công!")

        # 2. PP-OCRv6 (PaddleOCR)
        from paddleocr import PaddleOCR
        self.ppocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="vi"
        )
        print("[+] Khởi tạo PP-OCRv6 (PaddleOCR) thành công!")

        # Regex nhận biết các mẫu chuỗi thuần số, mã ID, ngày tháng, mã vạch
        self.code_pattern = re.compile(r"^[0-9\s\.\,\-\/\:\#\*\+\(\)]+$")
        self.alphanumeric_pattern = re.compile(r"^[A-Z0-9\-\/\.\:\#]+$")
        self.vietnamese_vowels = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ")

    def recognize_cell(self, patch_bgr: np.ndarray):
        """
        Nhận diện chữ trong 1 ô bằng VietOCR và PP-OCRv6, sau đó dung hợp thông minh.
        
        Returns:
            dict chứa:
            - final_text: Chuỗi văn bản tối ưu nhất
            - chosen_model: 'VietOCR' hoặc 'PP-OCRv6'
            - vietocr_text: Kết quả VietOCR
            - ppocr_text: Kết quả PP-OCRv6
            - confidence: Độ tin cậy ước lượng
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return {"final_text": "", "chosen_model": "None", "vietocr_text": "", "ppocr_text": "", "confidence": 0.0}

        # 1. Chạy VietOCR
        viet_text = ""
        try:
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(patch_rgb)
            viet_text = self.vietocr_predictor.predict(pil_img).strip()
        except Exception as e:
            viet_text = ""

        # 2. Chạy PP-OCRv6
        ppocr_text = ""
        ppocr_conf = 0.0
        try:
            pp_res = list(self.ppocr_engine.predict(patch_bgr))
            texts = []
            scores = []
            for r in pp_res:
                if isinstance(r, dict):
                    texts.extend(r.get("rec_texts", []))
                    scores.extend(r.get("rec_scores", []))
            ppocr_text = " ".join(texts).strip()
            ppocr_conf = float(np.mean(scores)) if scores else 0.85
        except Exception as e:
            ppocr_text = ""
            ppocr_conf = 0.0

        # 3. Định tuyến và Dung hợp thông minh (Smart Arbitration)
        final_text = ""
        chosen_model = ""
        conf = 0.90

        # Heuristic 1: Nếu một bên rỗng, chọn bên có chữ
        if viet_text and not ppocr_text:
            final_text, chosen_model = viet_text, "VietOCR"
        elif ppocr_text and not viet_text:
            final_text, chosen_model = ppocr_text, "PP-OCRv6"
        elif not viet_text and not ppocr_text:
            final_text, chosen_model = "", "None"
        else:
            # Cả hai bên đều có chữ:
            clean_pp = ppocr_text.replace(" ", "")
            has_vi_vowel = any(c in self.vietnamese_vowels for c in viet_text)

            # A. Chuỗi thuần số, ngày tháng, mã barcode, serial, số điện thoại -> Chọn PP-OCRv6
            if self.code_pattern.fullmatch(clean_pp) or (self.alphanumeric_pattern.fullmatch(clean_pp) and len(clean_pp) >= 4 and not has_vi_vowel):
                final_text = ppocr_text
                chosen_model = "PP-OCRv6"
                conf = max(0.95, ppocr_conf)
            # B. Chuỗi có dấu tiếng Việt chuẩn xác -> Chọn VietOCR
            elif has_vi_vowel:
                final_text = viet_text
                chosen_model = "VietOCR"
                conf = 0.98
            # C. Chuỗi Latinh thông thường hoặc tiêu đề
            else:
                # Nếu VietOCR dài hơn hoặc chi tiết hơn
                if len(viet_text) >= len(ppocr_text):
                    final_text = viet_text
                    chosen_model = "VietOCR"
                else:
                    final_text = ppocr_text
                    chosen_model = "PP-OCRv6"
                conf = ppocr_conf

        return {
            "final_text": final_text,
            "chosen_model": chosen_model,
            "vietocr_text": viet_text,
            "ppocr_text": ppocr_text,
            "confidence": round(conf, 3)
        }


# ============================================================================
# 4. BỘ TRỰC QUAN HÓA VÀ XUẤT ẢNH GHÉP 2 TRONG 1 (DUAL-VIEW VISUALIZER)
# ============================================================================

class DualViewVisualizer:
    """
    Tạo và xuất 1 file ảnh gồm 2 ảnh đặt cạnh nhau (Side-by-Side):
    - Bên trái: Ảnh gốc với các ô đã được quét (vẽ bounding box, số thứ tự #i).
    - Bên phải: Ảnh canvas trắng thể hiện chữ viết ứng với từng ô đã quét
      (vẽ tại tọa độ tương ứng, render tiếng Việt chuẩn Unicode, kèm nhãn model).
    """
    def __init__(self):
        # Tìm font tiếng Việt chuẩn trong hệ thống Windows
        self.font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        self.font_path = next((f for f in self.font_candidates if os.path.exists(f)), "arial.ttf")

    def _get_font(self, size: int, bold: bool = False):
        try:
            return ImageFont.truetype(self.font_path, size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, draw, text: str, max_width: int, font):
        """Ngắt dòng tự động cho văn bản tiếng Việt vừa với chiều rộng ô."""
        words = text.split(" ")
        lines = []
        current = ""
        for w in words:
            test = f"{current} {w}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            tw = bbox[2] - bbox[0]
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    def create_dual_view_image(self, original_bgr: np.ndarray, cells_with_ocr: list, header_title: str = ""):
        """
        Ghép 2 ảnh: Trái (Ảnh gốc + ô quét) | Phải (Chữ viết ứng với từng ô).
        """
        h_orig, w_orig = original_bgr.shape[:2]

        # --------------------------------------------------------------------
        # 1. TẠO ẢNH BÊN TRÁI: ẢNH GỐC VẼ CÁC Ô ĐÃ ĐƯỢC QUÉT
        # --------------------------------------------------------------------
        left_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        left_draw = ImageDraw.Draw(left_pil)
        badge_font = self._get_font(size=14, bold=True)

        # Bảng màu sắc nét cho từng loại ô
        color_palette = [
            (0, 150, 255),    # Xanh dương / Cyan
            (16, 185, 129),   # Xanh lục emerald
            (245, 158, 11),   # Vàng cam amber
            (139, 92, 246),   # Tím violet
            (236, 72, 153),   # Hồng magenta
            (239, 68, 68),    # Đỏ cam
        ]

        for item in cells_with_ocr:
            order_id = item["order_id"]
            box = item["box"]
            color = color_palette[(order_id - 1) % len(color_palette)]
            x1, y1, x2, y2 = box

            # Vẽ viền hình chữ nhật bao quanh ô
            left_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Vẽ badge số thứ tự ô ở góc trên bên trái: #1, #2...
            badge_text = f"#{order_id}"
            t_bbox = left_draw.textbbox((0, 0), badge_text, font=badge_font)
            bw = t_bbox[2] - t_bbox[0] + 8
            bh = t_bbox[3] - t_bbox[1] + 6

            by1 = max(0, y1 - bh)
            by2 = by1 + bh
            bx1 = x1
            bx2 = min(w_orig, bx1 + bw)

            # Nền badge màu nổi bật + chữ trắng
            left_draw.rectangle([bx1, by1, bx2, by2], fill=color)
            left_draw.text((bx1 + 4, by1 + 2), badge_text, fill=(255, 255, 255), font=badge_font)

        # --------------------------------------------------------------------
        # 2. TẠO ẢNH BÊN PHẢI: CHỮ VIẾT ỨNG VỚI TỪNG Ô ĐÃ QUÉT
        # --------------------------------------------------------------------
        right_pil = Image.new("RGB", (w_orig, h_orig), color=(250, 252, 255))
        right_draw = ImageDraw.Draw(right_pil)

        for item in cells_with_ocr:
            order_id = item["order_id"]
            box = item["box"]
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            color = color_palette[(order_id - 1) % len(color_palette)]

            text = item.get("final_text", "").strip()
            model_used = item.get("chosen_model", "")

            # A. Vẽ ô nền nhẹ tại tọa độ tương ứng với ảnh bên trái
            right_draw.rectangle([x1, y1, x2, y2], fill=(241, 245, 249), outline=color, width=2)

            # B. Badge số thứ tự và nhãn model
            model_tag = "VietOCR" if model_used == "VietOCR" else "PP-OCR"
            tag_text = f"#{order_id} [{model_tag}]" if model_used != "None" else f"#{order_id}"
            t_bbox = right_draw.textbbox((0, 0), tag_text, font=badge_font)
            badge_w = t_bbox[2] - t_bbox[0] + 8
            badge_h = t_bbox[3] - t_bbox[1] + 4

            # Kiểm tra va chạm với ô phía trên để đặt tab thông minh
            tab_y1 = y1 - badge_h
            tab_y2 = y1
            tab_x1 = x1
            tab_x2 = min(w_orig, x1 + badge_w)

            overlaps_other = tab_y1 < 0
            if not overlaps_other:
                for other in cells_with_ocr:
                    if other["order_id"] == order_id:
                        continue
                    ox1, oy1, ox2, oy2 = other["box"]
                    if not (tab_x2 <= ox1 or tab_x1 >= ox2 or tab_y2 <= oy1 or tab_y1 >= oy2):
                        overlaps_other = True
                        break

            if overlaps_other and bh >= badge_h + 16:
                # Đặt tab ở góc phải trên bên trong ô
                tab_y1 = y1 + 1
                tab_y2 = y1 + 1 + badge_h
                tab_x1 = max(x1 + 1, x2 - badge_w - 2)
                tab_x2 = min(w_orig, tab_x1 + badge_w)
            elif overlaps_other:
                tab_y1 = max(0, y1 - badge_h)
                tab_y2 = tab_y1 + badge_h

            right_draw.rectangle([tab_x1, tab_y1, tab_x2, tab_y2], fill=color)
            right_draw.text((tab_x1 + 4, tab_y1 + 1), tag_text, fill=(255, 255, 255), font=badge_font)

            # C. Render văn bản tiếng Việt bên trong ô
            if text:
                # Tính toán kích cỡ font phù hợp với chiều cao và chiều rộng của ô
                init_size = max(11, min(int(bh * 0.60), 28))
                font = self._get_font(size=init_size)
                lines = self._wrap_text(right_draw, text, max_width=bw - 8, font=font)

                # Thu nhỏ font nếu nhiều dòng không vừa chiều cao ô
                line_height = int(init_size * 1.35)
                total_text_h = len(lines) * line_height
                while total_text_h > bh - 4 and init_size > 10:
                    init_size -= 1
                    font = self._get_font(size=init_size)
                    line_height = int(init_size * 1.35)
                    lines = self._wrap_text(right_draw, text, max_width=bw - 8, font=font)
                    total_text_h = len(lines) * line_height

                # Căn giữa theo chiều dọc trong ô
                start_y = y1 + max(2, (bh - total_text_h) // 2)
                for line_idx, line_str in enumerate(lines):
                    curr_y = start_y + line_idx * line_height
                    if curr_y + line_height <= y2 + 4:
                        right_draw.text((x1 + 5, curr_y), line_str, fill=(15, 23, 42), font=font)

        # --------------------------------------------------------------------
        # 3. GHÉP 2 ẢNH VÀ THÊM THANH TIÊU ĐỀ HEADER CHUYÊN NGHIỆP
        # --------------------------------------------------------------------
        banner_h = 90
        divider_w = 6
        total_w = w_orig * 2 + divider_w
        total_h = h_orig + banner_h

        composite = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        comp_draw = ImageDraw.Draw(composite)

        # A. Vẽ Banner Tiêu Đề trên cùng
        # Nửa bên trái: Dark Slate Blue
        comp_draw.rectangle([0, 0, w_orig, banner_h], fill=(30, 41, 59))
        # Nửa bên phải: Dark Emerald Teal
        comp_draw.rectangle([w_orig + divider_w, 0, total_w, banner_h], fill=(15, 118, 110))
        # Vạch phân cách giữa
        comp_draw.rectangle([w_orig, 0, w_orig + divider_w, total_h], fill=(100, 116, 139))

        # Phông chữ tiêu đề
        title_font = self._get_font(size=22, bold=True)
        sub_font = self._get_font(size=14, bold=False)

        # Nội dung tiêu đề bên trái
        comp_draw.text((25, 16), "[BÊN TRÁI: ẢNH GỐC CÁC Ô ĐÃ ĐƯỢC QUÉT]", fill=(255, 255, 255), font=title_font)
        comp_draw.text((25, 52), "Phân đoạn: PP-DocLayoutV3 + PP-StructureV3 (Khung bố cục & Ô chữ)", fill=(148, 163, 184), font=sub_font)

        # Nội dung tiêu đề bên phải
        comp_draw.text((w_orig + divider_w + 25, 16), "[BÊN PHẢI: CHỮ VIẾT ỨNG VỚI TỪNG Ô ĐÃ QUÉT]", fill=(255, 255, 255), font=title_font)
        comp_draw.text((w_orig + divider_w + 25, 52), "Nhận diện chữ: VietOCR (Tiếng Việt) + PP-OCRv6 (Số / Mã ID)", fill=(167, 243, 208), font=sub_font)

        # B. Dán 2 ảnh trái và phải vào vị trí tương ứng
        composite.paste(left_pil, (0, banner_h))
        composite.paste(right_pil, (w_orig + divider_w, banner_h))

        return composite


# ============================================================================
# 5. PIPELINE THỰC THI HOÀN CHỈNH (MAIN CONTROLLER)
# ============================================================================

def run_document_dual_ocr_pipeline(
    image_path: str,
    output_dir: str = "output",
    vietocr_weights: str = None
):
    """
    Quy trình tích hợp:
    1. PP-DocLayoutV3 + PP-StructureV3 chia khung và ô.
    2. VietOCR + PP-OCRv6 nhận diện chữ từng ô.
    3. Xuất 1 file ảnh gồm 2 ảnh bên trái và bên phải.
    """
    if not os.path.exists(image_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(base_dir, os.path.basename(image_path))
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")

    stem_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 75)
    print(f"# KHỞI CHẠY PIPELINE OCR DUO-VIEW: {os.path.basename(image_path)}")
    print("#" * 75)

    # 1. BƯỚC 1: CHIA KHUNG VÀ Ô BẰNG PP-DOCLAYOUTV3 + PP-STRUCTUREV3
    segmenter = DocumentStructureSegmenter()
    processed_bgr, frames, cells = segmenter.segment(image_path)

    # 2. BƯỚC 2: NHẬN DIỆN CHỮ CHO TỪNG Ô VỚI VIETOCR + PP-OCRV6
    recognizer = DualOCRRecognizer(vietocr_weights_path=vietocr_weights)

    print("\n" + "=" * 80)
    print(f"{'Ô':<5} | {'BBOX [x1,y1,x2,y2]':<22} | {'MODEL CHỌN':<12} | {'KẾT QUẢ NHẬN DIỆN'}")
    print("=" * 80)

    cells_with_ocr = []
    for cell in cells:
        order_id = cell["order_id"]
        box = cell["box"]

        # Cắt ô ảnh kèm padding an toàn
        patch, padded_box = crop_image_patch(processed_bgr, box, pad_x=6, pad_y=4)
        if patch is None or patch.size == 0:
            continue

        # Nhận diện chữ
        ocr_result = recognizer.recognize_cell(patch)

        cell_data = {
            "order_id": order_id,
            "box": box,
            "padded_box": padded_box,
            "frame_label": cell.get("frame_label", "text"),
            "final_text": ocr_result["final_text"],
            "chosen_model": ocr_result["chosen_model"],
            "vietocr_text": ocr_result["vietocr_text"],
            "ppocr_text": ocr_result["ppocr_text"],
            "confidence": ocr_result["confidence"]
        }
        cells_with_ocr.append(cell_data)

        # In kết quả trực quan ra màn hình Console
        box_str = f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
        print(f"#{order_id:<4} | {box_str:<22} | {cell_data['chosen_model']:<12} | {cell_data['final_text']}")

    print("=" * 80)

    # 3. BƯỚC 3: XUẤT 1 FILE ẢNH GỒM 2 ẢNH GHÉP NGANG
    print("\n3. ĐANG VẼ VÀ XUẤT ẢNH GHÉP 2 TRONG 1 (SIDE-BY-SIDE DUAL VIEW)...")
    visualizer = DualViewVisualizer()
    dual_view_image = visualizer.create_dual_view_image(
        original_bgr=processed_bgr,
        cells_with_ocr=cells_with_ocr,
        header_title=f"Tài liệu: {os.path.basename(image_path)}"
    )

    output_img_path = os.path.join(output_dir, f"{stem_name}_dual_view.jpg")
    dual_view_image.save(output_img_path, quality=95)

    # 4. XUẤT DỮ LIỆU JSON VÀ MARKDOWN ĐỂ PHỤC VỤ TÍCH HỢP HỆ THỐNG
    output_json_path = os.path.join(output_dir, f"{stem_name}_dual_ocr.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "total_frames": len(frames),
            "total_cells": len(cells_with_ocr),
            "layout_frames": frames,
            "scanned_cells": cells_with_ocr
        }, f, ensure_ascii=False, indent=2)

    output_md_path = os.path.join(output_dir, f"{stem_name}_transcription.md")
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Kết quả OCR: {os.path.basename(image_path)}\n\n")
        f.write(f"- **Tổng số khung bố cục:** {len(frames)}\n")
        f.write(f"- **Tổng số ô đã quét và nhận diện:** {len(cells_with_ocr)}\n\n")
        f.write("| Ô | Khung Bố Cục | Model | Tọa Độ Box | Nội Dung Nhận Diện |\n")
        f.write("|---|---|---|---|---|\n")
        for c in cells_with_ocr:
            f.write(f"| #{c['order_id']} | `{c['frame_label']}` | {c['chosen_model']} | `{c['box']}` | {c['final_text']} |\n")

    print(f"\n[+] HOÀN THÀNH XUẤT SẮC!")
    print(f" [1] FILE ẢNH GHÉP 2 BÊN: {output_img_path}")
    print(f" [2] File dữ liệu JSON:   {output_json_path}")
    print(f" [3] File Markdown:       {output_md_path}")
    print("=" * 80 + "\n")

    return output_img_path, output_json_path


# ============================================================================
# 6. ĐIỂM KHỞI CHẠY (CLI ENTRYPOINT)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline OCR: PP-DocLayoutV3 + PP-StructureV3 -> VietOCR + PP-OCRv6 -> Dual-View Image")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh tài liệu")
    parser.add_argument("--output", type=str, default="output", help="Thư mục xuất kết quả")
    parser.add_argument("--vietocr-weights", type=str, default=None, help="Đường dẫn file weights của VietOCR (.pth)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.image:
        target_image = args.image
    else:
        # Danh sách các ảnh mẫu trong thư mục
        candidates = [
            os.path.join(base_dir, "test_document.png"),
            os.path.join(base_dir, "test_doc.jpg"),
            os.path.join(base_dir, "test_ngang.jpg"),
            "OCR_Project/test_document.png"
        ]
        target_image = next((c for c in candidates if os.path.exists(c)), "OCR_Project/test_document.png")

    run_document_dual_ocr_pipeline(
        image_path=target_image,
        output_dir=args.output,
        vietocr_weights=args.vietocr_weights
    )
