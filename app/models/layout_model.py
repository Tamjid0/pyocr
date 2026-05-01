import os
from surya.model.detection.segformer import load_processor, load_model
from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.layout import batch_layout_detection
from surya.ordering import batch_ordering
from PIL import Image

from transformers import VisionEncoderDecoderConfig
from surya.model.ordering.decoder import MBART_ATTENTION_CLASSES

# Monkeypatch for Surya 0.4.14 / Transformers compatibility
MBART_ATTENTION_CLASSES["sdpa"] = MBART_ATTENTION_CLASSES["eager"]

import gc
import torch
import logging

logger = logging.getLogger(__name__)

class LayoutModel:
    def __init__(self):
        # We no longer load models at startup to save RAM
        pass

    def detect_layout(self, image: Image.Image):
        """Loads Layout model, detects regions, unloads model."""
        logger.info("    -> Loading Layout model into RAM...")
        model = load_model()
        processor = load_processor()
        
        layout_predictions = batch_layout_detection([image], model, processor)
        
        logger.info("    -> Unloading Layout model from RAM...")
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return layout_predictions[0]

    def detect_order(self, image: Image.Image, bboxes: list):
        """Loads Order model, detects reading order, unloads model."""
        logger.info("    -> Loading Ordering model into RAM...")
        order_model = load_order_model()
        order_processor = load_order_processor()
        
        order_predictions = batch_ordering([image], [bboxes], order_model, order_processor)
        
        logger.info("    -> Unloading Ordering model from RAM...")
        del order_model
        del order_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return order_predictions[0]
