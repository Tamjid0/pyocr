from paddleocr import PaddleOCR
import numpy as np

class OCRModel:
    def __init__(self):
        # use_angle_cls=True allows detecting text orientation
        # use_gpu=False for Cloud Run CPU compatibility
        self.engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

    def extract_text(self, image_np: np.ndarray):
        """Runs OCR on an image (or sub-region) and returns structured results."""
        result = self.engine.ocr(image_np, cls=True)
        return result[0] if result else []
