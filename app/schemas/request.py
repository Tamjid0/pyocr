from pydantic import BaseModel, Field
from typing import Optional

class ProcessingFlags(BaseModel):
    extract_text: bool = True
    extract_math: bool = False
    detect_layout: bool = True

class OCRRequest(BaseModel):
    document_id: str = Field(..., description="UUID or Hash of the document")
    page_index: int = Field(..., description="0-based page index")
    image_data: str = Field(..., description="Base64 encoded PNG or JPEG image")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    processing_flags: Optional[ProcessingFlags] = Field(default_factory=ProcessingFlags)

class OCRPageItem(BaseModel):
    page_index: int = Field(..., description="0-based page index")
    image_data: str = Field(..., description="Base64 encoded PNG or JPEG image")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")

class OCRBatchRequest(BaseModel):
    document_id: str = Field(..., description="UUID or Hash of the document")
    pages: list[OCRPageItem] = Field(..., description="List of pages to process as a batch")
    processing_flags: Optional[ProcessingFlags] = Field(default_factory=ProcessingFlags)

