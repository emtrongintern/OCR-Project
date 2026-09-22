import os
import sys
import shutil
from pathlib import Path

# Cấu hình UTF-8 cho Windows Terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def create_sample_dataset_yaml(output_path: str = "dataset_layout.yaml"):
    """
    Tạo file cấu hình dữ liệu mẫu chuẩn YOLO.
    Người dùng chỉ cần sửa đường dẫn thư mục images/labels tương ứng.
    """
    sample_yaml = """# File cấu hình tập dữ liệu cho YOLO Layout Detection
path: ./dataset_layout      # Thư mục gốc chứa dataset
train: images/train         # Thư mục ảnh huấn luyện
val: images/val             # Thư mục ảnh kiểm thử (validation)

# Danh sách các lớp (classes) tương ứng với các trường trong tài liệu
names:
  0: id             # Trường số ID / CCCD / Mã số (được route vào PP-OCRv6)
  1: ho_ten         # Trường Họ và tên (được route vào VietOCR)
  2: ngay_sinh      # Trường ngày sinh (được route vào PP-OCRv6)
  3: dia_chi        # Trường địa chỉ (được route vào VietOCR)
  4: que_quan       # Trường quê quán (được route vào VietOCR)
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sample_yaml.strip())
    print(f"[Dataset] Đã tạo file mẫu cấu hình dữ liệu: {output_path}")


def train_yolo_layout(
    data_yaml: str = "dataset_layout.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    target_weights_name: str = "yolo26.pt"
):
    """
    Khởi chạy quá trình Fine-tune YOLO với các tham số tối ưu cho tài liệu.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Lỗi] Vui lòng cài đặt ultralytics: pip install ultralytics")
        return

    if not os.path.exists(data_yaml):
        print(f"[Cảnh báo] Chưa tìm thấy file '{data_yaml}'. Đang tạo file mẫu...")
        create_sample_dataset_yaml(data_yaml)
        print(f"[Hướng dẫn] Vui lòng gán nhãn dữ liệu (bằng LabelImg hoặc Roboflow)")
        print(f"            và cập nhật đúng đường dẫn trong file '{data_yaml}' trước khi train.")
        return

    print("=" * 70)
    print("BẮT ĐẦU HUẤN LUYỆN YOLO CHO BÀI TOÁN TÀI LIỆU CỐ ĐỊNH (CHỐNG LỆCH IN)")
    print("=" * 70)
    print(f" - Model nền:       {base_model}")
    print(f" - Cấu hình Data:   {data_yaml}")
    print(f" - Số epochs:       {epochs}")
    print(f" - Kích thước ảnh:  {imgsz}")
    print(f" - Batch size:      {batch_size}")
    print("=" * 70)

    # Nạp mô hình nền
    model = YOLO(base_model)

    # Huấn luyện với các tham số Data Augmentation chống lệch in
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        # --- Cấu hình Augmentation đặc trị cho lỗi "In hơi lệch" ---
        translate=0.08,    # Dịch chuyển ngẫu nhiên ±8% tọa độ
        degrees=3.0,       # Xoay nghiêng nhẹ ±3 độ
        scale=0.05,        # Co dãn kích thước nhẹ ±5%
        shear=1.0,         # Biến dạng góc nhẹ
        perspective=0.0005,# Góc phối cảnh scan nhẹ
        mosaic=0.2,        # Giảm mosaic để tránh xé nát văn bản tài liệu
        # -----------------------------------------------------------
        name="yolo_layout_exp"
    )

    # Copy file weights tốt nhất ra thư mục làm việc thành yolo26.pt
    best_weights = Path("runs/detect/yolo_layout_exp/weights/best.pt")
    if best_weights.exists():
        destination = Path(__file__).parent / target_weights_name
        shutil.copy(best_weights, destination)
        print("=" * 70)
        print(f"[THÀNH CÔNG] Đã lưu file weights fine-tuned tốt nhất vào: {destination}")
        print(f"Bây giờ bạn có thể chạy: python newdocument_parsing.py")
        print("=" * 70)
    else:
        print("[Hoàn tất] Quá trình train đã xong. Kiểm tra thư mục runs/detect để lấy file best.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune YOLO Layout cho tài liệu in lệch")
    parser.add_argument("--data", type=str, default="dataset_layout.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Mô hình nền (yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train")
    parser.add_argument("--init-yaml", action="store_true", help="Chỉ tạo file data.yaml mẫu")
    args = parser.parse_args()

    if args.init_yaml:
        create_sample_dataset_yaml(args.data)
    else:
        train_yolo_layout(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz
        )
