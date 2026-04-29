from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.model_manager import models
from app.api.endpoints import router as api_router
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    logger.info("Starting up Document Perception Engine...")
    models.load_models()
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
