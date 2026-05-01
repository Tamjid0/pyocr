import requests
import base64
import fitz # PyMuPDF
import io
from PIL import Image
import json

def test_ocr():
    pdf_path = "test-files/unnamed.pdf"
    url = "http://localhost:8080/api/v1/ocr/process-page"
    
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    
    img = Image.open(io.BytesIO(img_data))
    width, height = img.size
    
    base64_image = base64.b64encode(img_data).decode("utf-8")
    
    payload = {
        "document_id": "test_doc",
        "page_index": 0,
        "image_data": base64_image,
        "image_width": width,
        "image_height": height,
        "processing_flags": {
            "extract_math": True,
            "extract_tables": True,
            "refine_ordering": True
        }
    }
    
    print("Sending request to OCR service...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        print("Success! Received response.")
        
        # Save result to a file for inspection
        with open("test-files/test_result.json", "w") as f:
            json.dump(result, f, indent=2)
            
        print(f"Result saved to test-files/test_result.json")
        
        # Summary of results
        regions = result.get("extracted_regions", [])
        print(f"\nExtracted {len(regions)} regions:")
        for r in regions:
            print(f" - [{r['region_type']}] Confidence: {r['confidence_score']:.2f}, Lines: {len(r['extracted_lines'])}")
            if r['extracted_lines']:
                print(f"   Snippet: {r['extracted_lines'][0]['text'][:50]}...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ocr()
