import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Cấu hình UTF-8 cho Windows Terminal
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

try:
    from ultralytics import YOLO
except ImportError:
    print("[Lỗi] Thư viện ultralytics chưa được cài đặt. Chạy: pip install ultralytics")
    sys.exit(1)


def read_image_safe(image_path: str) -> np.ndarray:
    """Đọc ảnh an toàn trên Windows hỗ trợ đường dẫn tiếng Việt có dấu."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Không thể giải mã file ảnh: {image_path}")
    return img_bgr


def save_image_safe(image_path: str, img_bgr: np.ndarray) -> bool:
    """Lưu ảnh an toàn trên Windows hỗ trợ đường dẫn tiếng Việt có dấu."""
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    ext = os.path.splitext(image_path)[1]
    if not ext:
        ext = ".jpg"
        image_path += ext
    success, encoded_img = cv2.imencode(ext, img_bgr)
    if success:
        with open(image_path, "wb") as f:
            encoded_img.tofile(f)
        return True
    return False


# ==============================================================================
# PHƯƠNG THỨC 1: CẮT BOUNDING BOX THÔNG THƯỜNG CÓ PADDING (SAFE-PADDING CROP)
# Phù hợp cho nhận diện Layout tài liệu in lệch, tránh cắt sát mép mất nét chữ
# ==============================================================================
def crop_standard_bbox(
    img_bgr: np.ndarray,
    box_xyxy: list,
    padding_x: int = 8,
    padding_y: int = 4
) -> tuple:
    """
    Cắt vùng ảnh theo Bounding Box [x1, y1, x2, y2] kèm đệm viền an toàn (padding).
    
    Returns:
        (cropped_image, [x1_pad, y1_pad, x2_pad, y2_pad])
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(coord)) for coord in box_xyxy]

    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(w, x2 + padding_x)
    y2_pad = min(h, y2 + padding_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, [x1, y1, x2, y2]

    crop = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    return crop, [x1_pad, y1_pad, x2_pad, y2_pad]


# ==============================================================================
# PHƯƠNG THỨC 2: CẮT THEO HỘP XOAY (ORIENTED BOUNDING BOX - OBB) & XOAY THẲNG
# Áp dụng cho yolo26-obb / yolo26x-obb.pt khi tài liệu hoặc dòng chữ bị scan nghiêng
# ==============================================================================
def crop_oriented_bbox(
    img_bgr: np.ndarray,
    pts_4x2: np.ndarray
) -> np.ndarray:
    """
    Cắt và xoay thẳng vùng ảnh theo 4 điểm góc của hộp chữ nhật xoay (OBB).
    Sử dụng thuật toán phối cảnh Perspective Transform (Warp Perspective).
    
    Args:
        img_bgr: Mảng ảnh BGR gốc.
        pts_4x2: Mảng 4 điểm tọa độ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    Returns:
        Ảnh cắt đã được xoay thẳng về phương nằm ngang chuẩn.
    """
    pts = np.array(pts_4x2, dtype="float32")
    
    # Sắp xếp 4 điểm theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-Left (tổng x+y nhỏ nhất)
    rect[2] = pts[np.argmax(s)]       # Bottom-Right (tổng x+y lớn nhất)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-Right (hiệu y-x nhỏ nhất)
    rect[3] = pts[np.argmax(diff)]    # Bottom-Left (hiệu y-x lớn nhất)

    (tl, tr, br, bl) = rect

    # Tính chiều rộng thực tế của hình chữ nhật
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(round(width_a)), int(round(width_b)))

    # Tính chiều cao thực tế của hình chữ nhật
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(round(height_a)), int(round(height_b)))

    if max_w <= 0 or max_h <= 0:
        return None

    # Tọa độ đích sau khi xoay thẳng thành hình chữ nhật góc 0 độ
    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    # Ma trận biến đổi và làm phẳng (Warp Perspective)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, matrix, (max_w, max_h))
    return warped


# ==============================================================================
# QUY TRÌNH TỰ ĐỘNG CẮT ẢNH BẰNG YOLO26
# ==============================================================================
def crop_image_with_yolo26(
    image_path: str,
    model_path: str = "yolo26n.pt",
    output_dir: str = "output/crops",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    padding_x: int = 8,
    padding_y: int = 4,
    save_crops: bool = True
) -> list:
    """
    Phát hiện và cắt tất cả các vùng đối tượng/trường văn bản bằng YOLO26.
    Tự động nhận biết mô hình là Detection (BBox chuẩn) hay OBB (Hộp xoay).

    Returns:
        Danh sách các dict chứa thông tin chi tiết từng phần cắt:
        [
            {
                "crop_id": 1,
                "label": "ho_ten" / "id" / ...,
                "confidence": 0.95,
                "cropped_image": np.ndarray,
                "saved_path": str or None,
                "box_or_pts": list
            }, ...
        ]
    """
    print("=" * 70)
    print(f"BẮT ĐẦU CẮT ẢNH BẰNG MÔ HÌNH: {model_path}")
    print(f"Ảnh nguồn: {image_path}")
    print("=" * 70)

    # 1. Đọc ảnh
    img_bgr = read_image_safe(image_path)
    img_stem = Path(image_path).stem

    # 2. Khởi tạo mô hình YOLO26
    print(f"[YOLO26] Đang nạp weights từ: {model_path}...")
    model = YOLO(model_path)

    # 3. Dự đoán
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )
    result = results[0]

    cropped_results = []
    crop_counter = 0

    # KIỂM TRA TASK CỦA MODEL (OBB hay BBOX CHUẨN)
    is_obb = hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0
    names = result.names or {}

    if is_obb:
        print(f"[YOLO26-OBB] Phát hiện {len(result.obb)} vùng xoay (Oriented Bounding Boxes).")
        obbs = result.obb
        for i in range(len(obbs)):
            cls_id = int(obbs.cls[i].item())
            label = names.get(cls_id, f"class_{cls_id}")
            conf = float(obbs.conf[i].item())
            pts_4x2 = obbs.xyxyxyxy[i].cpu().numpy()

            crop_patch = crop_oriented_bbox(img_bgr, pts_4x2)
            if crop_patch is None or crop_patch.size == 0:
                continue

            crop_counter += 1
            saved_path = None
            if save_crops:
                save_filename = f"{img_stem}_obb_{crop_counter:03d}_{label}.jpg"
                saved_path = os.path.join(output_dir, label, save_filename)
                save_image_safe(saved_path, crop_patch)

            cropped_results.append({
                "crop_id": crop_counter,
                "label": label,
                "confidence": round(conf, 4),
                "cropped_image": crop_patch,
                "saved_path": saved_path,
                "box_or_pts": pts_4x2.tolist()
            })

    elif result.boxes is not None and len(result.boxes) > 0:
        print(f"[YOLO26-Detect] Phát hiện {len(result.boxes)} vùng hộp chữ nhật (Bounding Boxes).")
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().tolist()

            crop_patch, padded_box = crop_standard_bbox(
                img_bgr, xyxy, padding_x=padding_x, padding_y=padding_y
            )
            if crop_patch is None or crop_patch.size == 0:
                continue

            crop_counter += 1
            saved_path = None
            if save_crops:
                save_filename = f"{img_stem}_crop_{crop_counter:03d}_{label}.jpg"
                saved_path = os.path.join(output_dir, label, save_filename)
                save_image_safe(saved_path, crop_patch)

            cropped_results.append({
                "crop_id": crop_counter,
                "label": label,
                "confidence": round(conf, 4),
                "cropped_image": crop_patch,
                "saved_path": saved_path,
                "box_or_pts": padded_box
            })
    else:
        print("[YOLO26] Không phát hiện thấy đối tượng nào phù hợp với ngưỡng tin cậy.")

    print(f"[HOÀN THÀNH] Đã cắt thành công {len(cropped_results)} phần ảnh.")
    if save_crops and cropped_results:
        print(f"Các ảnh cắt đã được lưu vào thư mục: {os.path.abspath(output_dir)}")
    print("=" * 70)

    return cropped_results


# ==============================================================================
# GIAO DIỆN DÒNG LỆNH (CLI)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Công cụ cắt ảnh tự động chuyên nghiệp bằng YOLO26 (Hỗ trợ cả BBox chuẩn & OBB xoay nghiêng)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="Datasets/Testcases/test_document.png",
        help="Đường dẫn file ảnh tài liệu đầu vào"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="OCR_Project/yolo26x-obb.pt",
        help="Đường dẫn weights model YOLO26 (.pt) - ví dụ: yolo26n.pt hoặc OCR_Project/yolo26x-obb.pt"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/yolo26_crops",
        help="Thư mục xuất các file ảnh cắt"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng độ tin cậy confidence threshold (mặc định: 0.25)"
    )
    parser.add_argument(
        "--pad-x",
        type=int,
        default=8,
        help="Padding bề ngang (pixel) chống lệch in khi cắt BBox (mặc định: 8)"
    )
    parser.add_argument(
        "--pad-y",
        type=int,
        default=4,
        help="Padding bề dọc (pixel) chống mất nét chữ khi cắt BBox (mặc định: 4)"
    )

    args = parser.parse_args()

    # Kiểm tra đường dẫn dự phòng
    model_path = args.model
    if not os.path.exists(model_path):
        if os.path.exists("yolo26n.pt"):
            model_path = "yolo26n.pt"
        elif os.path.exists("OCR_Project/yolo26x-obb.pt"):
            model_path = "OCR_Project/yolo26x-obb.pt"

    crop_image_with_yolo26(
        image_path=args.image,
        model_path=model_path,
        output_dir=args.output,
        conf_thresh=args.conf,
        padding_x=args.pad_x,
        padding_y=args.pad_y,
        save_crops=True
    )


if __name__ == "__main__":
    main()
