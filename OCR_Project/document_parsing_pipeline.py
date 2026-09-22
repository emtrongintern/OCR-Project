import os
import sys
import json
from pathlib import Path
from paddleocr import PPStructureV3

# Cấu hình encoding UTF-8 cho Terminal Windows
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


def build_content_tree(res, page_index=1):
    """
    Tạo cấu trúc cây các khối nội dung từ kết quả PP-StructureV3 kết hợp PP-DocLayoutV3.
    """
    input_path = res.get("input_path", "Tài liệu không xác định")
    width = res.get("width", 0)
    height = res.get("height", 0)
    
    # Thông tin xoay / tiền xử lý tài liệu
    angle = -1
    if res.get("model_settings", {}).get("use_doc_preprocessor"):
        angle = res.get("doc_preprocessor_res", {}).get("angle", -1)

    parsing_blocks = res.get("parsing_res_list", [])
    
    tree_lines = []
    tree_lines.append(f"📄 [TRANG {page_index}] {os.path.basename(str(input_path))} (Kích thước: {width}x{height}px | Góc xoay: {angle}°)")
    tree_lines.append(f"├── 🤖 Model cấu hình: PP-StructureV3 + PP-DocLayoutV3 (Layout Detection)")
    tree_lines.append(f"├── 📊 Tổng số khối nội dung phân tích: {len(parsing_blocks)}")
    tree_lines.append(f"└── 🗂️ Cấu trúc cây thứ tự đọc (Reading Order Tree):")

    def format_single_block(block, idx, is_last, indent="    "):
        branch = "└── " if is_last else "├── "
        sub_indent = indent + ("    " if is_last else "│   ")

        # Thuộc tính của LayoutBlock hoặc dict
        if hasattr(block, "to_dict"):
            label = getattr(block, "label", "unknown")
            bbox = getattr(block, "bbox", [])
            order = getattr(block, "order_index", idx + 1)
            content = getattr(block, "content", "")
            child_blocks = getattr(block, "child_blocks", [])
            block_id = getattr(block, "index", idx)
        elif isinstance(block, dict):
            label = block.get("block_label", "unknown")
            bbox = block.get("block_bbox", [])
            order = block.get("block_order", idx + 1)
            content = block.get("block_content", "")
            child_blocks = block.get("child_blocks", [])
            block_id = block.get("block_id", idx)
        else:
            label = str(type(block))
            bbox = []
            order = idx + 1
            content = str(block)
            child_blocks = []
            block_id = idx

        # Biểu tượng tương ứng với loại khối
        icon_map = {
            "doc_title": "👑 [TIÊU ĐỀ CHÍNH]",
            "paragraph_title": "📌 [TIÊU ĐỀ ĐOẠN]",
            "text": "📝 [VĂN BẢN]",
            "table": "📊 [BẢNG BIỂU]",
            "image": "🖼️ [HÌNH ẢNH]",
            "header_image": "🖼️ [ẢNH HEADER]",
            "formula": "📐 [CÔNG THỨC TOÁN]",
            "seal": "🔴 [CON DẤU]",
            "chart": "📈 [BIỂU ĐỒ]",
            "header": "🔝 [HEADER]",
            "footer": "🔚 [FOOTER]",
            "footnote": "📎 [CHÚ THÍCH]",
            "vision_footnote": "📎 [CHÚ THÍCH]",
            "number": "🔢 [SỐ]",
            "reference": "📚 [TÀI LIỆU THAM KHẢO]",
            "aside_text": "💬 [VĂN BẢN BÊN LỀ]",
        }
        type_str = icon_map.get(label, f"🏷️ [{label.upper()}]")

        lines = []
        lines.append(f"{indent}{branch}[Khối #{order}] {type_str} (ID: {block_id})")
        lines.append(f"{sub_indent}├── 📍 BBox: {bbox}")

        # Hiển thị nội dung khối
        content_clean = str(content).strip().replace("\r\n", " ").replace("\n", " ")
        if len(content_clean) > 80:
            content_clean = content_clean[:80] + "..."
        if content_clean:
            lines.append(f"{sub_indent}└── 💬 Nội dung: \"{content_clean}\"")
        else:
            lines.append(f"{sub_indent}└── 💬 Nội dung: (Trống / Dạng đối tượng ảnh)")

        # Nếu có khối con (child blocks)
        if child_blocks:
            lines.append(f"{sub_indent}└── 📂 Khối con lồng ghép ({len(child_blocks)} sub-blocks):")
            for c_idx, c_block in enumerate(child_blocks):
                c_is_last = (c_idx == len(child_blocks) - 1)
                c_lines = format_single_block(c_block, c_idx, c_is_last, indent=sub_indent + "    ")
                lines.extend(c_lines)

        return lines

    for idx, block in enumerate(parsing_blocks):
        is_last = (idx == len(parsing_blocks) - 1)
        tree_lines.extend(format_single_block(block, idx, is_last))

    return "\n".join(tree_lines)


def run_document_parsing(image_path: str, output_dir: str = "output"):
    """
    Khởi tạo và thực thi pipeline PP-StructureV3 kết hợp PP-DocLayoutV3,
    sau đó in cấu trúc cây ra Terminal và lưu vào thư mục output.
    """
    if not os.path.exists(image_path):
        # Thử tìm tương đối theo thư mục file
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file tài liệu tại: {image_path}")

    print("=" * 70)
    print("1. ĐANG KHỞI TẠO PIPELINE PP-STRUCTUREV3 KẾT HỢP PP-DOCLAYOUTV3...")
    print("=" * 70)

    # Kết hợp model PP-DocLayoutV3 vào pipeline PP-StructureV3
    pipeline = PPStructureV3(
        layout_detection_model_name="PP-DocLayoutV3",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
    )

    print(f"\n2. ĐANG PHÂN TÍCH TÀI LIỆU: {image_path}...")
    results = pipeline.predict(input=image_path)

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    all_tree_texts = []

    print("\n" + "=" * 70)
    print("3. CẤU TRÚC CÂY CÁC KHỐI NỘI DUNG (HIERARCHICAL CONTENT BLOCK TREE)")
    print("=" * 70 + "\n")

    for i, res in enumerate(results):
        page_num = i + 1
        
        # 1. In cấu trúc cây trực tiếp ra màn hình Terminal
        tree_text = build_content_tree(res, page_index=page_num)
        print(tree_text)
        print("\n" + "-" * 70 + "\n")
        all_tree_texts.append(tree_text)

        # 2. In cấu trúc chi tiết thông qua res.print() của PaddleX
        print(f"[PaddleX Native Output - Trang {page_num}]:")
        res.print()
        print("\n" + "=" * 70 + "\n")

        # 3. Lưu kết quả chuẩn của PaddleX (JSON, Markdown, Bounding Box Images)
        res.save_to_json(save_path=output_dir)
        res.save_to_markdown(save_path=output_dir)
        res.save_to_img(save_path=output_dir)

    # 4. Lưu toàn bộ cấu trúc cây ra file text và file json riêng biệt trong thư mục output
    tree_file_path = os.path.join(output_dir, f"{base_name}_tree_structure.txt")
    with open(tree_file_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_tree_texts))

    print(f"4. ĐÃ LƯU CÁC KẾT QUẢ VÀO THƯ MỤC '{output_dir}':")
    print(f"   [+] File cây cấu trúc văn bản: {tree_file_path}")
    print(f"   [+] File JSON chi tiết:        {os.path.join(output_dir, f'{base_name}_res.json')}")
    print(f"   [+] File Markdown trích xuất:  {os.path.join(output_dir, f'{base_name}.md')}")
    print(f"   [+] Ảnh vẽ khung phân tích:    {output_dir}/{base_name}_*.jpg/png")
    print("=" * 70)
    print("HOÀN THÀNH BƯỚC PHÂN ĐOẠN CẤU TRÚC!")
    print("=" * 70)

    # Tích hợp xuất ảnh ghép 2 trong 1: Bố cục ô quét (trái) và Chữ viết nhận diện (phải)
    try:
        from structure_vietocr_dual_view import run_document_dual_ocr_pipeline
        print("\n" + "=" * 70)
        print("5. TIẾP TỤC: NHẬN DIỆN CHỮ (VIETOCR + PP-OCRV6) & XUẤT ẢNH DUAL-VIEW...")
        print("=" * 70)
        run_document_dual_ocr_pipeline(image_path=image_path, output_dir=output_dir)
    except Exception as e:
        print(f"[Lưu ý] Không chạy tiếp bước Dual OCR do: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Document Parsing Pipeline: PP-DocLayoutV3 + PP-StructureV3 + VietOCR")
    parser.add_argument("image", nargs="?", default=None, help="Đường dẫn file ảnh tài liệu")
    parser.add_argument("--output", default="output", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    if args.image:
        sample_image = args.image
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "test_document.png"),
            os.path.join(base_dir, "test_ngang.jpg"),
            os.path.join(base_dir, "test_doc.jpg"),
        ]
        sample_image = next((c for c in candidates if os.path.exists(c)), "OCR_Project/test_document.png")

    run_document_parsing(sample_image, output_dir=args.output)
