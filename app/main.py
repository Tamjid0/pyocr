import logging
import os

# Configure Logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.model_manager import models
from app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    logger.info("🚀 Starting up Document Perception Engine (PRODUCTION / A10 Mode)...")
    
    # --- A10 CO-TENANCY: Cap Surya to 6GB out of 24GB total VRAM ---
    # MinerU (procr) will use the remaining ~18GB for its massive KV Cache.
    import torch
    if torch.cuda.is_available():
        # --- A10 CO-TENANCY: Cap Surya to 6GB explicitly ---
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        fraction = min(1.0, 6.0 / total_memory_gb)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        logger.info(f"🛡️ VRAM Cap: Surya limited to 6GB ({fraction*100:.1f}% of {total_memory_gb:.1f}GB).")
        
        # --- PYTORCH HARDWARE ACCELERATION ---
        # Note: cudnn.benchmark is disabled because dynamic image sizes cause massive compilation lag
        torch.backends.cuda.matmul.allow_tf32 = True  
        torch.backends.cudnn.allow_tf32 = True
        logger.info("⚡ PyTorch Hardware Acceleration (TF32) ENABLED")
    
    models.load_models()
    logger.info("✅ Startup complete. Service is ready.")
    yield
    logger.info("Shutting down Document Perception Engine...")

app = FastAPI(
    title="Document Perception Engine",
    description="Python OCR Microservice for Structured Document Understanding",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": models.initialized}

@app.get("/diagnostic")
async def diagnostic():
    import torch
    import psutil
    
    cuda_available = torch.cuda.is_available()
    cuda_info = {}
    if cuda_available:
        cuda_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
        }
        
    return {
        "status": "ready",
        "models_initialized": models.initialized,
        "cuda": {
            "available": cuda_available,
            **cuda_info
        },
        "system_ram": f"{psutil.virtual_memory().available / 1024**2:.2f} MB available",
        "persistent_state": {
            "layout": models.layout._layout_model is not None if models.layout else False,
            "ocr": models.ocr._rec_model is not None if models.ocr else False
        }
    }
