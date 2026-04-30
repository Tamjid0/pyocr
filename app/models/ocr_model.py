from typing import List
from PIL import Image
from surya.ocr import run_recognition
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import logging

logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        logger.info("  - Initializing Surya Recognition model (Stable PyTorch)...")
        self.model = load_rec_model()
        self.processor = load_rec_processor()

    def extract_text(self, image: Image.Image, langs: List[str] = None):
        """Runs Surya OCR on an image and returns structured results."""
        if langs is None:
            langs = ["en"]
        # run_recognition returns List[OCRResult]
        results = run_recognition([image], [langs], self.model, self.processor)
        return results[0]
