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

class LayoutModel:
    def __init__(self):
        # Force eager attention implementation for compatibility
        self.model = load_model()
        self.processor = load_processor()
        
        # Explicitly load and fix ordering model config
        self.order_model = load_order_model()
        self.order_processor = load_order_processor()

    def detect_layout(self, image: Image.Image):
        """Detects regions and their types."""
        # batch_layout_detection returns a list of LayoutResult
        layout_predictions = batch_layout_detection([image], self.model, self.processor)
        return layout_predictions[0]

    def detect_order(self, image: Image.Image, bboxes: list):
        """Detects reading order of provided bboxes."""
        # batch_ordering returns a list of OrderResult
        order_predictions = batch_ordering([image], [bboxes], self.order_model, self.order_processor)
        return order_predictions[0]
