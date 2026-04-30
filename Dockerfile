FROM python:3.10-slim

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch
RUN pip install --no-cache-dir --default-timeout=10000 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

# Pre-download Surya models
COPY download_models.py .
RUN python download_models.py

ENV VERSION=1.0.6
COPY . .

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
