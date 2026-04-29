import base64
import io
import numpy as np
from PIL import Image

def decode_base64_image(base64_str: str) -> Image.Image:
    """Converts base64 string to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Converts PIL Image to OpenCV-compatible numpy array."""
    return np.array(image)[:, :, ::-1] # RGB to BGR for PaddleOCR

def get_sub_image(image: Image.Image, bbox: list) -> Image.Image:
    """Crops a PIL image based on [x0, y0, x1, y1] bbox."""
    return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
