import logging
import os

# Configure Logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Force eager attention to prevent 'sdpa' KeyError in older Surya versions
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.model_manager import models
from app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    logger.info("🚀 Starting up Document Perception Engine...")
    logger.info("🛠️ Configuration: Super-Lazy Mode (Models and Libraries load on first request).")
    models.load_models()
    logger.info("✅ Startup complete. Service is ready.")
    yield
    # Shutdown: Clean up if needed
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
