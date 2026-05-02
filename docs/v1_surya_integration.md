# PyOCR v1 - Surya Integration Documentation

## Overview
PyOCR is a high-performance Python microservice designed to provide structured OCR capabilities to the Antigravity (pdfx) platform. It leverages the **Surya** OCR library for layout analysis, line detection, and text recognition.

## Architecture

### 1. Model Manager (`app/core/model_manager.py`)
- **Singleton Pattern**: Ensures only one instance of the model manager exists.
- **Eager Loading**: All models (Layout, Detection, Recognition) are loaded into VRAM during the FastAPI startup sequence.
- **VRAM Persistence**: Models remain in memory between requests to eliminate the 60-second "cold start" penalty.

### 2. Models
- **Layout Model**: Detects document regions (Paragraphs, Headers, Tables, Math).
- **OCR Model**: 
    - **Detection**: Performs global line-level bounding box detection.
    - **Recognition**: Uses GPU batching (size: 32) to convert line crops into text.
- **Math Model**: Specialized handling for LaTeX extraction.

### 3. Perception Pipeline (`app/services/pipeline.py`)
The `process_page` method follows a high-fidelity 6-step sequence:
1. **Decode**: Base64 image decoding.
2. **Layout Detection**: Identifies logical blocks (paragraphs).
3. **Reading Order**: Sorts layout blocks logically.
4. **Global Line Detection**: Detects every line on the page independently of layout blocks to avoid "garbage text" repetition.
5. **Global Recognition**: Recognizes all detected lines in a single batched GPU pass.
6. **Spatial Mapping**: Performs a spatial join (overlap > 50%) to assign recognized lines back to their parent layout blocks.

## Performance Optimizations
- **Forced GPU**: Explicitly detects and forces `cuda` device for all models.
- **Batched Recognition**: Set to 32 lines per pass for maximum GPU throughput.
- **Eager VRAM**: Models are pre-loaded at boot, reducing subsequent request latency to ~5-8 seconds per page.
- **Profiling**: Every request logs a detailed breakdown: `Total`, `Decode`, `Layout`, `Order`, `Det`, `Rec`, `Map`.

## API Endpoints

### `POST /api/v1/ocr/process-page`
- **Input**: `OCRRequest` (JSON) containing base64 image data.
- **Output**: `OCRResponse` containing hierarchical region and line data with bounding boxes.

### `GET /diagnostic`
Returns the current health and hardware status:
- CUDA availability and device name.
- VRAM allocation and reservation.
- Persistent state of individual models.
- System RAM availability.

## Integration with Node.js
- **Service**: `PythonOCRService.ts` in the Node server handles communication via the Ngrok tunnel.
- **Injection**: `DocumentProcessor.ts` injects the line-level nodes into the `DocumentGraph`.
- **Citations**: High-fidelity line nodes allow the AI to cite specific lines using ⟦n⟧ aliases.
