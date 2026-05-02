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
            logger.info("📦 Beginning STABLE model manager initialization (Eager Loading)...")
            
            logger.info("⚙️ Initializing Layout Manager...")
            self.layout = LayoutModel()
            
            logger.info("⚙️ Initializing Recognition Manager...")
            self.ocr = OCRModel()
            
            logger.info("⚙️ Initializing Math Manager...")
            self.math = MathModel(self.ocr)

            # --- EAGER LOADING ---
            logger.info("🚀 Eagerly loading models into VRAM to eliminate first-request latency...")
            self.layout._load_layout()
            self.ocr._load_detection()
            self.ocr._load_recognition()
            
            self.initialized = True
            logger.info("🌟 ALL MODELS LOADED AND READY IN VRAM")
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Failed to load models: {str(e)}")
            raise e

models = ModelManager()
