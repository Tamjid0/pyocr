from pydantic import BaseModel, Field
from typing import List, Optional

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class TextStyle(BaseModel):
    font_size: float = Field(..., description="Derived from bbox height or native font size")
    is_bold: bool = Field(False, description="Required for Heading Isolation in Node.js")

class ExtractedWord(BaseModel):
    text: str
    bbox: BBox
    confidence_score: Optional[float] = 0.0

class ExtractedLine(BaseModel):
    text: str
    bbox: BBox
    style: TextStyle
    confidence_score: Optional[float] = 0.0
    extracted_words: List[ExtractedWord] = Field(default_factory=list)

class ExtractedRegion(BaseModel):
    region_index: int
    region_type: str = Field(..., description="Text, Math, Table, Image, Figure")
    bbox: BBox
    confidence_score: Optional[float] = 0.0
    extracted_lines: List[ExtractedLine] = Field(default_factory=list)

class OCRResponse(BaseModel):
    page_index: int
    page_width: int
    page_height: int
    reading_order_hints: List[int] = Field(..., description="Array of region indices representing Surya's detected reading order")
    extracted_regions: List[ExtractedRegion] = Field(default_factory=list)

class OCRBatchResponse(BaseModel):
    document_id: str
    results: List[OCRResponse]

