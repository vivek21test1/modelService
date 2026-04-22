# ── Base: PyTorch 2.1 + CUDA 11.8 + Python 3.10 ─────────────────────────────
# EasyOCR requires PyTorch and CUDA. This image ships both, avoiding a
# separate torch install that would balloon build time significantly.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System libs required by OpenCV / EasyOCR (libglib, libGL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
