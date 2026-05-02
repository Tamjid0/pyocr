# PyOCR - Document Perception Engine

PyOCR is the Python-based vision intelligence layer for the Antigravity platform. It provides high-fidelity OCR, layout analysis, and mathematical formula extraction using the Surya library.

## Documentation
- [v1 - Surya Integration](docs/v1_surya_integration.md): Detailed architecture and performance specs.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Start the service: `uvicorn app.main:app --port 8080`
3. Verify: `GET http://localhost:8080/diagnostic`

## Features
- **GPU Accelerated**: Native CUDA support with batched inference.
- **Persistent VRAM**: Models stay loaded for <10s response times.
- **Hierarchical Output**: Returns structured regions and precise line-level coordinates.
- **Math Extraction**: Built-in support for LaTeX and technical symbols.
