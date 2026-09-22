# from paddleocr import PaddleOCR

# # Default: PP-OCRv6_medium
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )
# sample_image="OCR_Project/general_ocr_002.png"
# result = ocr.predict(sample_image)
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_path = "OCR_Project/test_ngang.jpg"
model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("OCR_Project/test_ngang.jpg", stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
for result in results:
    for idx, (score, label_id, box, polygon_points) in enumerate(zip(result["scores"], result["labels"], result["boxes"], result["polygon_points"])):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label]}, score: {score:.2f}, box: {box}, polygon_points: {polygon_points}")
