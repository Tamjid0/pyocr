import logging
from app.models.layout_model import LayoutModel
from app.models.ocr_model import OCRModel
from app.models.math_model import MathModel

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.layout = None
            cls._instance.ocr = None
            cls._instance.math = None
        return cls._instance

    def load_models(self):
        if self.initialized:
            return

        try:
            logger.info("📦 Beginning STABLE model initialization sequence...")
            
            logger.info("🔍 Loading Surya Layout Detection model...")
            self.layout = LayoutModel()
            
            logger.info("🔍 Loading Surya Recognition engine...")
            self.ocr = OCRModel()
            
            logger.info("🔍 Initializing Math extraction (Surya-based)...")
            self.math = MathModel(self.ocr)
            
            self.initialized = True
            logger.info("🌟 ALL MODELS LOADED SUCCESSFULLY")
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Failed to load models: {str(e)}")
            raise e

models = ModelManager()
