import os
# Force eager attention for Surya during download
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

from surya.model.ordering.decoder import MBART_ATTENTION_CLASSES
# Monkeypatch for Surya 0.4.14 / Transformers compatibility
MBART_ATTENTION_CLASSES["sdpa"] = MBART_ATTENTION_CLASSES["eager"]

from surya.model.detection.segformer import load_model, load_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

def download():
    print("📥 Pre-downloading Surya STABLE models...")
    load_model()
    load_processor()
    load_order_model()
    load_order_processor()
    load_rec_model()
    load_rec_processor()
    print("✅ All Surya models downloaded.")

if __name__ == "__main__":
    download()
