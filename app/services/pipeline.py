import logging
import os
from datetime import datetime
from app.core.model_manager import models
from app.schemas.request import OCRRequest, OCRBatchRequest
from app.schemas.response import OCRResponse, ExtractedRegion, ExtractedLine, ExtractedWord, BBox, TextStyle, OCRBatchResponse
from app.utils.image_utils import decode_base64_image, pil_to_numpy, get_sub_image
import numpy as np

logger = logging.getLogger(__name__)

_STATS_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pyocr_stats.log")

def _write_stats(line: str):
    try:
        with open(_STATS_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

class PerceptionPipeline:
    @staticmethod
    async def process_page(request: OCRRequest) -> OCRResponse:
        import time
        start_total = time.perf_counter()
        logger.info(f"Processing page {request.page_index} for document {request.document_id}")
        
        # 1. Decode Image
        t0 = time.perf_counter()
        image = decode_base64_image(request.image_data)
        t_decode = time.perf_counter() - t0
        
        # 2. Surya Layout Detection
        t0 = time.perf_counter()
        layout = models.layout.detect_layout(image)
        region_bboxes = [region.bbox for region in layout.bboxes]
        t_layout = time.perf_counter() - t0
        
        # 3. Surya Reading Order
        t0 = time.perf_counter()
        order = models.layout.detect_order(image, region_bboxes)
        positions = [(i, b.position) for i, b in enumerate(order.bboxes)]
        sorted_positions = sorted(positions, key=lambda x: x[1])
        reading_order_hints = [p[0] for p in sorted_positions]
        t_order = time.perf_counter() - t0
        
        # 4. Global Detection (Detect ALL lines on the page)
        t0 = time.perf_counter()
        det_results = models.ocr.detect_lines([image])
        page_lines = det_results[0].bboxes 
        t_det = time.perf_counter() - t0
        
        # 5. Global Recognition (Recognize ALL lines on the page)
        t0 = time.perf_counter()
        recognition_bboxes = [[line.bbox for line in page_lines]]
        ocr_results = models.ocr.extract_text([image], bboxes=recognition_bboxes, langs=["en"])
        ocr_lines = ocr_results[0].text_lines
        t_rec = time.perf_counter() - t0
        
        # 6. Spatial Assignment (Map Lines -> Regions)
        t0 = time.perf_counter()
        extracted_regions = []
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

        for line in ocr_lines:
            line_bbox = line.bbox
            best_region_idx = -1
            max_overlap = 0
            
            for idx, region in enumerate(layout.bboxes):
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
                    style=TextStyle(font_size=line_bbox[3] - line_bbox[1], is_bold=is_bold or (target_region.region_type == "Math")),
                    confidence_score=line.confidence,
                    extracted_words=[]
                ))
        t_map = time.perf_counter() - t0
        
        total_time = time.perf_counter() - start_total
        logger.info(f"⏱️ PERFORMANCE [{request.document_id}]: Total {total_time:.2f}s | Decode {t_decode:.2f}s | Layout {t_layout:.2f}s | Order {t_order:.2f}s | Det {t_det:.2f}s | Rec {t_rec:.2f}s | Map {t_map:.2f}s")
        _write_stats(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PAGE {request.page_index} | Total: {total_time:.2f}s | Layout: {t_layout:.2f}s | Order: {t_order:.2f}s | Det: {t_det:.2f}s | Rec: {t_rec:.2f}s | Map: {t_map:.2f}s")

        return OCRResponse(
            page_index=request.page_index,
            page_width=image.width,
            page_height=image.height,
            reading_order_hints=reading_order_hints,
            extracted_regions=extracted_regions
        )

    @staticmethod
    async def process_batch(request: OCRBatchRequest) -> OCRBatchResponse:
        import time
        start_total = time.perf_counter()
        logger.info(f"Processing bulk batch of {len(request.pages)} pages for document {request.document_id}")
        
        if not request.pages:
            return OCRBatchResponse(document_id=request.document_id, results=[])

        # 1. Decode all images
        t0 = time.perf_counter()
        images = []
        for page in request.pages:
            images.append(decode_base64_image(page.image_data))
        t_decode = time.perf_counter() - t0
        
        # 2. Batch Layout Detection
        t0 = time.perf_counter()
        from surya.layout import batch_layout_detection
        layout_model, layout_processor = models.layout._load_layout()
        layout_predictions = batch_layout_detection(images, layout_model, layout_processor)
        t_layout = time.perf_counter() - t0
        
        # 3. Batch Reading Order
        t0 = time.perf_counter()
        from surya.ordering import batch_ordering
        order_model, order_processor = models.layout._load_order()
        all_region_bboxes = [[region.bbox for region in page_layout.bboxes] for page_layout in layout_predictions]
        order_predictions = batch_ordering(images, all_region_bboxes, order_model, order_processor)
        t_order = time.perf_counter() - t0
        
        # 4. Batch Detection (All lines in all images)
        t0 = time.perf_counter()
        det_results = models.ocr.detect_lines(images, batch_size=16)
        t_det = time.perf_counter() - t0
        
        # 5. Batch Recognition
        t0 = time.perf_counter()
        recognition_bboxes = [[line.bbox for line in page_det.bboxes] for page_det in det_results]
        ocr_results = models.ocr.extract_text(images, bboxes=recognition_bboxes, langs=["en"], batch_size=32)
        t_rec = time.perf_counter() - t0
        
        # 6. Spatial Assignment & Output Formatting for each page
        t0 = time.perf_counter()
        results = []
        for img_idx, page in enumerate(request.pages):
            image = images[img_idx]
            layout = layout_predictions[img_idx]
            order = order_predictions[img_idx]
            ocr_lines = ocr_results[img_idx].text_lines
            
            # Reconstruct reading order hints
            positions = [(i, b.position) for i, b in enumerate(order.bboxes)]
            sorted_positions = sorted(positions, key=lambda x: x[1])
            reading_order_hints = [p[0] for p in sorted_positions]
            
            extracted_regions = []
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

            for line in ocr_lines:
                line_bbox = line.bbox
                best_region_idx = -1
                max_overlap = 0
                
                for idx, region in enumerate(layout.bboxes):
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
                        style=TextStyle(font_size=line_bbox[3] - line_bbox[1], is_bold=is_bold or (target_region.region_type == "Math")),
                        confidence_score=line.confidence,
                        extracted_words=[]
                    ))
            
            results.append(OCRResponse(
                page_index=page.page_index,
                page_width=image.width,
                page_height=image.height,
                reading_order_hints=reading_order_hints,
                extracted_regions=extracted_regions
            ))
            
        t_map = time.perf_counter() - t0
        total_time = time.perf_counter() - start_total
        logger.info(f"⏱️ BATCH PERFORMANCE [{request.document_id}]: Total {total_time:.2f}s | Decode {t_decode:.2f}s | Layout {t_layout:.2f}s | Order {t_order:.2f}s | Det {t_det:.2f}s | Rec {t_rec:.2f}s | Map {t_map:.2f}s")
        _write_stats(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BATCH {len(request.pages)}pg | Total: {total_time:.2f}s | Layout: {t_layout:.2f}s | Order: {t_order:.2f}s | Det: {t_det:.2f}s | Rec: {t_rec:.2f}s | Map: {t_map:.2f}s")
        
        return OCRBatchResponse(
            document_id=request.document_id,
            results=results
        )

