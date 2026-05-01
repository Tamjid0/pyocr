from typing import List
from PIL import Image
import logging
import gc
import torch

logger = logging.getLogger(__name__)

class OCRModel:
    def __init__(self):
        # We no longer load models at startup to save RAM
        pass

    def extract_text(self, images: List[Image.Image], langs: List[str] = None):
        """Runs Surya OCR on a list of images and returns structured results."""
        from surya.ocr import run_recognition
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        
        if langs is None:
            langs = ["en"]
            
        logger.info("    -> Loading Recognition model into RAM...")
        model = load_rec_model()
        processor = load_rec_processor()
        
        lang_list = [langs for _ in images]
        # Surya requires explicit bboxes. Since each image is already a cropped
        # region, pass a single bbox covering the full dimensions of each image.
        bboxes = [[[0, 0, img.width, img.height]] for img in images]
        results = run_recognition(images, lang_list, model, processor, bboxes=bboxes)
        
        logger.info("    -> Unloading Recognition model from RAM...")
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
