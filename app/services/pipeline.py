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
        region_bboxes = [region.bbox for region in layout.bboxes]
        
        # 3. Surya Reading Order
        order = models.layout.detect_order(image, region_bboxes)
        positions = [(i, b.position) for i, b in enumerate(order.bboxes)]
        sorted_positions = sorted(positions, key=lambda x: x[1])
        reading_order_hints = [p[0] for p in sorted_positions]
        
        # 4. Global Detection (Detect ALL lines on the page)
        # This fixes the "garbage text" issue by ensuring the OCR engine sees line-level crops, not paragraph-level crops.
        det_results = models.ocr.detect_lines([image])
        page_lines = det_results[0].bboxes # List of detected line bboxes
        
        # 5. Global Recognition (Recognize ALL lines on the page)
        # Convert surya bboxes to the format expected by run_recognition
        recognition_bboxes = [[line.bbox for line in page_lines]]
        ocr_results = models.ocr.extract_text([image], bboxes=recognition_bboxes, langs=["en"])
        ocr_lines = ocr_results[0].text_lines # List of TextLine objects
        
        # 6. Spatial Assignment (Map Lines -> Regions)
        extracted_regions = []
        
        # Initialize regions with empty line lists
        for idx, region in enumerate(layout.bboxes):
            region_type = region.label
            contract_type = "Text"
            if "Math" in region_type: contract_type = "Math"
            elif "Table" in region_type: contract_type = "Table"
            elif "Figure" in region_type or "Image" in region_type: contract_type = "Figure"
            
            extracted_regions.append(ExtractedRegion(
                region_index=idx,
                region_type=contract_type,
                bbox=BBox(x0=region.bbox[0], y0=region.bbox[1], x1=region.bbox[2], y1=region.bbox[3]),
                confidence_score=region.confidence,
                extracted_lines=[]
            ))

        # Assign each detected line to the region it most overlaps with
        for line in ocr_lines:
            line_bbox = line.bbox # [x0, y0, x1, y1]
            
            # Find best matching region
            best_region_idx = -1
            max_overlap = 0
            
            for idx, region in enumerate(layout.bboxes):
                # Calculate intersection area
                r_bbox = region.bbox
                ix0 = max(line_bbox[0], r_bbox[0])
                iy0 = max(line_bbox[1], r_bbox[1])
                ix1 = min(line_bbox[2], r_bbox[2])
                iy1 = min(line_bbox[3], r_bbox[3])
                
                if ix1 > ix0 and iy1 > iy0:
                    overlap = (ix1 - ix0) * (iy1 - iy0)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_region_idx = idx
            
            if best_region_idx != -1:
                target_region = extracted_regions[best_region_idx]
                is_bold = any(kw in layout.bboxes[best_region_idx].label.lower() for kw in ['title', 'header', 'section'])
                
                target_region.extracted_lines.append(ExtractedLine(
                    text=line.text,
                    bbox=BBox(x0=line_bbox[0], y0=line_bbox[1], x1=line_bbox[2], y1=line_bbox[3]),
                    style=TextStyle(
                        font_size=line_bbox[3] - line_bbox[1], 
                        is_bold=is_bold or (target_region.region_type == "Math")
                    ),
                    confidence_score=line.confidence,
                    extracted_words=[]
                ))

        return OCRResponse(
            page_index=request.page_index,
            page_width=image.width,
            page_height=image.height,
            reading_order_hints=reading_order_hints,
            extracted_regions=extracted_regions
        )
