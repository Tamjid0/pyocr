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
        
        # 2. Surya Layout Detection
        layout = models.layout.detect_layout(image)
        bboxes = [region.bbox for region in layout.bboxes]
        
        # 3. Surya Reading Order
        order = models.layout.detect_order(image, bboxes)
        positions = [(i, b.position) for i, b in enumerate(order.bboxes)]
        sorted_positions = sorted(positions, key=lambda x: x[1])
        reading_order_hints = [p[0] for p in sorted_positions]
        
        extracted_regions = []
        
        # 4. Region-based Extraction
        for idx, region in enumerate(layout.bboxes):
            region_type = region.label
            bbox = region.bbox
            
            contract_type = "Text"
            if "Math" in region_type: contract_type = "Math"
            elif "Table" in region_type: contract_type = "Table"
            elif "Figure" in region_type or "Image" in region_type: contract_type = "Figure"
            
            is_bold = any(kw in region_type.lower() for kw in ['title', 'header', 'section'])
            region_lines = []
            
            # Using Surya for everything now for stability
            sub_img = get_sub_image(image, bbox)
            ocr_result = models.ocr.extract_text(sub_img, langs=["en"])
            
            for line in ocr_result.text_lines:
                # Surya bbox is [x0, y0, x1, y1] relative to sub-image
                line_bbox_rel = line.bbox
                line_text = line.text
                line_conf = line.confidence
                
                # Convert relative sub-image coords to absolute page coords
                abs_x0 = bbox[0] + line_bbox_rel[0]
                abs_y0 = bbox[1] + line_bbox_rel[1]
                abs_x1 = bbox[0] + line_bbox_rel[2]
                abs_y1 = bbox[1] + line_bbox_rel[3]
                
                region_lines.append(ExtractedLine(
                    text=line_text,
                    bbox=BBox(x0=abs_x0, y0=abs_y0, x1=abs_x1, y1=abs_y1),
                    style=TextStyle(font_size=abs_y1 - abs_y0, is_bold=is_bold or (contract_type == "Math")),
                    confidence_score=line_conf,
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
            page_width=image.width,
            page_height=image.height,
            reading_order_hints=reading_order_hints,
            extracted_regions=extracted_regions
        )
