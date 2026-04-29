from app.models.layout_model import LayoutModel
from app.models.ocr_model import OCRModel
from app.models.math_model import MathModel
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.layout = None
        self.ocr = None
        self.math = None
        self.initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        """Initializes all models. Call this during FastAPI lifespan startup."""
        if self.initialized:
            return
            
        logger.info("Initializing Perception Models (Surya, PaddleOCR, Pix2Text)...")
        try:
            self.layout = LayoutModel()
            self.ocr = OCRModel()
            self.math = MathModel()
            self.initialized = True
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise e

# Global singleton
models = ModelManager.get_instance()
