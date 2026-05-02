from typing import List
from PIL import Image
import logging
import gc
import torch

logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        self._rec_model = None
        self._rec_processor = None
        self._det_model = None
        self._det_processor = None

    def _load_recognition(self):
        if self._rec_model is None:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            logger.info("    -> Loading Recognition model into VRAM (Persistent)...")
            self._rec_model = load_rec_model()
            self._rec_processor = load_rec_processor()
        return self._rec_model, self._rec_processor

    def _load_detection(self):
        if self._det_model is None:
            from surya.model.detection.segformer import load_model as load_det_model
            from surya.model.detection.segformer import load_processor as load_det_processor
            logger.info("    -> Loading Detection model into VRAM (Persistent)...")
            self._det_model = load_det_model()
            self._det_processor = load_det_processor()
        return self._det_model, self._det_processor

    def detect_lines(self, images: List[Image.Image]):
        """Detects text lines in a list of images."""
        from surya.detection import batch_text_detection
        model, processor = self._load_detection()
        det_predictions = batch_text_detection(images, model, processor)
        return det_predictions

    def extract_text(self, images: List[Image.Image], bboxes: List[List[List[float]]], langs: List[str] = None):
        """Runs Surya OCR on a list of images and specific bboxes."""
        from surya.ocr import run_recognition
        
        if langs is None:
            langs = ["en"]
            
        model, processor = self._load_recognition()
        lang_list = [langs for _ in images]
        
        results = run_recognition(images, lang_list, model, processor, bboxes=bboxes)
        return results
