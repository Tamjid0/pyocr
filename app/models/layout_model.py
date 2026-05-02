import os
from PIL import Image
import gc
import torch
import logging

logger = logging.getLogger(__name__)

class LayoutModel:
    def __init__(self):
        self._layout_model = None
        self._layout_processor = None
        self._order_model = None
        self._order_processor = None
        self._device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🚀 LayoutModel initialized with device: {self._device}")

    def _load_layout(self):
        if self._layout_model is None:
            from surya.model.detection.segformer import load_processor, load_model
            logger.info(f"    -> Loading Layout model into {self._device} (Persistent)...")
            self._layout_model = load_model(device=self._device)
            self._layout_processor = load_processor(device=self._device)
        return self._layout_model, self._layout_processor

    def _load_order(self):
        if self._order_model is None:
            from surya.model.ordering.model import load_model as load_order_model
            from surya.model.ordering.processor import load_processor as load_order_processor
            from surya.model.ordering.decoder import MBART_ATTENTION_CLASSES
            
            # Monkeypatch for Surya 0.4.14 / Transformers compatibility
            MBART_ATTENTION_CLASSES["sdpa"] = MBART_ATTENTION_CLASSES["eager"]
            
            logger.info(f"    -> Loading Ordering model into {self._device} (Persistent)...")
            self._order_model = load_order_model(device=self._device)
            self._order_processor = load_order_processor(device=self._device)
        return self._order_model, self._order_processor

    def detect_layout(self, image: Image.Image):
        """Persistent layout detection."""
        from surya.layout import batch_layout_detection
        model, processor = self._load_layout()
        layout_predictions = batch_layout_detection([image], model, processor)
        return layout_predictions[0]

    def detect_order(self, image: Image.Image, bboxes: list):
        """Persistent reading order detection."""
        from surya.ordering import batch_ordering
        model, processor = self._load_order()
        order_predictions = batch_ordering([image], [bboxes], model, processor)
        return order_predictions[0]
