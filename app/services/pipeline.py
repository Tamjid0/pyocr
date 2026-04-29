import logging
from app.core.model_manager import models
from app.schemas.request import OCRRequest
from app.schemas.response import OCRResponse, ExtractedRegion, ExtractedLine, ExtractedWord, BBox, TextStyle
from app.utils.image_utils import decode_base64_image, pil_to_numpy, get_sub_image
import numpy as np

logger = logging.getLogger(__name__)

class PerceptionPipeline:
    @staticmethod
    async def process_page(request: OCRRequest) -> OCRResponse:
        logger.info(f"Processing page {request.page_index} for document {request.document_id}")
        
        # 1. Decode Image
        image = decode_base64_image(request.image_data)
        image_np = pil_to_numpy(image)
        
        # 2. Surya Layout Detection
        layout = models.layout.detect_layout(image)
        bboxes = [region.bbox for region in layout.bboxes]
        
        # 3. Surya Reading Order
        order = models.layout.detect_order(image, bboxes)
        # order.selection_indices gives us the sorted indices of regions
        reading_order_hints = order.selection_indices
        
        extracted_regions = []
        
        # 4. Region-based Extraction
        for idx, region in enumerate(layout.bboxes):
            region_type = region.label # e.g. 'Text', 'Math', 'Title'
            bbox = region.bbox # [x0, y0, x1, y1]
            
            # Map Surya labels to Contract labels
            contract_type = "Text"
            if "Math" in region_type: contract_type = "Math"
            elif "Table" in region_type: contract_type = "Table"
            elif "Figure" in region_type or "Image" in region_type: contract_type = "Figure"
            
            # Heuristic for is_bold
            is_bold = any(kw in region_type.lower() for kw in ['title', 'header', 'section'])
            
            region_lines = []
            
            if contract_type == "Math" and request.processing_flags.extract_math:
                # Specialized Math Extraction
                sub_img = get_sub_image(image, bbox)
                latex = models.math.extract_latex(sub_img)
                
                # Treat the whole math region as one line for simplicity
                region_lines.append(ExtractedLine(
                    text=latex,
                    bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                    style=TextStyle(font_size=bbox[3] - bbox[1], is_bold=True),
                    confidence_score=region.confidence,
                    extracted_words=[ExtractedWord(
                        text=latex,
                        bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                        confidence_score=region.confidence
                    )]
                ))
            else:
                # Standard Text Extraction (PaddleOCR)
                sub_img_np = pil_to_numpy(get_sub_image(image, bbox))
                ocr_results = models.ocr.extract_text(sub_img_np)
                
                for line_data in ocr_results:
                    # line_data format: [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence) ]
                    line_bbox_rel = line_data[0]
                    line_text = line_data[1][0]
                    line_conf = line_data[1][1]
                    
                    # Convert relative sub-image coords to absolute page coords
                    abs_x0 = bbox[0] + line_bbox_rel[0][0]
                    abs_y0 = bbox[1] + line_bbox_rel[0][1]
                    abs_x1 = bbox[0] + line_bbox_rel[2][0]
                    abs_y1 = bbox[1] + line_bbox_rel[2][1]
                    
                    region_lines.append(ExtractedLine(
                        text=line_text,
                        bbox=BBox(x0=abs_x0, y0=abs_y0, x1=abs_x1, y1=abs_y1),
                        style=TextStyle(font_size=abs_y1 - abs_y0, is_bold=is_bold),
                        confidence_score=line_conf,
                        # PaddleOCR line results don't easily split words without more processing
                        # For now, we'll treat the line as a single word if needed, 
                        # or we could perform a split. The contract allows empty words.
                        extracted_words=[] 
                    ))

            extracted_regions.append(ExtractedRegion(
                region_index=idx,
                region_type=contract_type,
                bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                confidence_score=region.confidence,
                extracted_lines=region_lines
            ))
            
        return OCRResponse(
            page_index=request.page_index,
            page_width=request.image_width,
            page_height=request.image_height,
            reading_order_hints=reading_order_hints,
            extracted_regions=extracted_regions
        )
