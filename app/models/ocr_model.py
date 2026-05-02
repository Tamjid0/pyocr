import logging
import gc
import torch
from typing import List
from PIL import Image

# Force GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        self._rec_model = None
        self._rec_processor = None
        self._det_model = None
        self._det_processor = None
        self._device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🚀 OCRModel initialized with device: {self._device}")

    def _load_recognition(self):
        if self._rec_model is None:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            logger.info(f"    -> Loading Recognition model into {self._device} (Persistent)...")
            self._rec_model = load_rec_model(device=self._device)
            self._rec_processor = load_rec_processor()
        return self._rec_model, self._rec_processor

    def _load_detection(self):
        if self._det_model is None:
            from surya.model.detection.segformer import load_model as load_det_model
            from surya.model.detection.segformer import load_processor as load_det_processor
            logger.info(f"    -> Loading Detection model into {self._device} (Persistent)...")
            self._det_model = load_det_model(device=self._device)
            self._det_processor = load_det_processor()
        return self._det_model, self._det_processor

    def detect_lines(self, images: List[Image.Image], batch_size: int = 16):
        """Detects text lines in a list of images."""
        from surya.detection import batch_text_detection
        model, processor = self._load_detection()
        det_predictions = batch_text_detection(images, model, processor, batch_size=batch_size)
        return det_predictions

    def extract_text(self, images: List[Image.Image], bboxes: List[List[List[float]]], langs: List[str] = None, batch_size: int = 32):
        """Runs Surya OCR on a list of images and specific bboxes."""
        from surya.ocr import run_recognition
        
        if langs is None:
            langs = ["en"]
            
        model, processor = self._load_recognition()
        lang_list = [langs for _ in images]
        
        results = run_recognition(images, lang_list, model, processor, bboxes=bboxes, batch_size=batch_size)
        return results
