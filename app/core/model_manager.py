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
            logger.info("📦 Beginning STABLE model manager initialization (Lazy Loading enabled)...")
            
            logger.info("⚙️ Initializing Layout Manager (Model will load on demand)...")
            self.layout = LayoutModel()
            
            logger.info("⚙️ Initializing Recognition Manager (Model will load on demand)...")
            self.ocr = OCRModel()
            
            logger.info("⚙️ Initializing Math Manager (Model will load on demand)...")
            self.math = MathModel(self.ocr)
            
            self.initialized = True
            logger.info("🌟 MODEL MANAGERS READY (RAM overhead: Minimal)")
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Failed to load models: {str(e)}")
            raise e

models = ModelManager()
