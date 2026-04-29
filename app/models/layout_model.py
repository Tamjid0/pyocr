import os
from surya.model.layout.processor import load_processor
from surya.model.layout.model import load_model
from surya.ocr import run_ocr
from surya.layout import run_layout
from surya.ordering import run_ordering
from PIL import Image

class LayoutModel:
    def __init__(self):
        self.model = load_model()
        self.processor = load_processor()
        # We also need ordering and detection models for full structural understanding
        from surya.model.ordering.model import load_model as load_order_model
        from surya.model.ordering.processor import load_processor as load_order_processor
        self.order_model = load_order_model()
        self.order_processor = load_order_processor()

    def detect_layout(self, image: Image.Image):
        """Detects regions and their types."""
        layout_predictions = run_layout([image], [self.model], [self.processor])
        return layout_predictions[0]

    def detect_order(self, image: Image.Image, bboxes: list):
        """Detects reading order of provided bboxes."""
        order_predictions = run_ordering([image], [bboxes], [self.order_model], [self.order_processor])
        return order_predictions[0]
