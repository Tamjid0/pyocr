from fastapi import APIRouter, HTTPException
from app.schemas.request import OCRRequest
from app.schemas.response import OCRResponse
from app.services.pipeline import PerceptionPipeline
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

import traceback

@router.post("/ocr/process-page", response_model=OCRResponse)
async def process_page(request: OCRRequest):
    """
    Main endpoint for processing a single document page.
    Follows /docs/python-ocr-contract.json strictly.
    """
    try:
        result = await PerceptionPipeline.process_page(request)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error processing page: {error_trace}")
        raise HTTPException(status_code=500, detail=error_trace)
