import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

# 1. Gọi cấu hình chuẩn của kiến trúc vgg_transformer
config = Cfg.load_config_from_name('vgg_transformer')

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
config['weights'] = os.path.join(base_dir, 'vgg_transformer.pth')
config['device'] = 'cpu'
detector = Predictor(config)
img_path = os.path.join(base_dir, 'test_ngang.jpg')
img = Image.open(img_path)
ket_qua = detector.predict(img)

print("VietOCR kết quả:", ket_qua)