from PIL import Image
import logging

logger = logging.getLogger(__name__)

class MathModel:
    def __init__(self, ocr_model):
        self.ocr_model = ocr_model

    def extract_latex(self, image: Image.Image):
        """Extracts text/math using Surya Recognition."""
        result = self.ocr_model.extract_text(image, langs=["en"])
        return " ".join([line.text for line in result.text_lines])
