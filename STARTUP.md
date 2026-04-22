# modelService — Startup Guide

---

## Option A — Run Locally

Use this to test EasyOCR on your own machine before deploying to Lightning.ai.

### Prerequisites

- Python 3.10
- (Optional) NVIDIA GPU with CUDA drivers — if not available, set `GPU=false` to run on CPU

### 1. Create a virtual environment

```bash
cd modelService
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> EasyOCR downloads model weights (~100 MB) on first run only.
> Cached at `~/.EasyOCR/model/` — not re-downloaded on restarts.

### 3. Create environment file

```bash
cp .env.example .env
```

For CPU-only local testing, edit `.env`:
```env
PORT=8000
GPU=false
LANGUAGES=["en"]
```

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
Loading EasyOCR model (gpu=False, languages=['en']) ...
EasyOCR model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 5. Test locally

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ready","model_loaded":true,"gpu":false}

# OCR from file
curl -X POST http://localhost:8000/ocr \
  -F "file=@/path/to/image.jpg"
# {"text":"extracted text here","lines":["..."]}

# OCR from base64
curl -X POST http://localhost:8000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<base64-string>"}'
```

### 6. Run with Docker locally (optional)

```bash
# Build
docker build -t modelservice .

# Run with GPU
docker run --gpus all -p 8000:8000 --env-file .env modelservice

# Run without GPU (CPU only)
docker run -p 8000:8000 -e GPU=false modelservice
```

---

## Option B — Deploy on Lightning.ai Studio

### Prerequisites

- Lightning.ai account with free-trial credit ($15)
- Python 3.10
- GPU Studio created and open (T4 recommended)

### 1. Upload code to the Studio

Inside the Studio terminal:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/modelService
```

Or use the Studio file browser to drag-and-drop the `modelService/` folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
Loading EasyOCR model (gpu=True, languages=['en']) ...
EasyOCR model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Expose the port and get the public URL

In the Studio UI → click **Expose Port** → enter `8000` → confirm.

Lightning.ai generates a public HTTPS URL like:
```
https://vivek21test1-get-gpu-project-ocr-service.lit.cloud
```

Copy this URL — it is your `LIGHTNING_STUDIO_URL_N` for the gateway `.env`.

### 5. Verify

```bash
curl https://your-studio.lit.cloud/health
# {"status":"ready","model_loaded":true,"gpu":true}

curl -X POST https://your-studio.lit.cloud/ocr \
  -F "file=@sample.jpg"
# {"text":"...extracted text...","lines":["..."]}
```

### 6. Configure auto-start (so server restarts on Studio wake)

In the Studio, open a Notebook → create a cell titled **On-start actions**:

```python
import subprocess
subprocess.Popen([
    "uvicorn", "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])
```

---

## Config Parameters

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the FastAPI server listens on |
| `GPU` | `true` | Set `false` for CPU-only local testing |
| `LANGUAGES` | `["en"]` | OCR languages e.g. `["en","hi"]` for English + Hindi |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns model status and GPU info |
| `POST` | `/ocr` | Upload image file (JPEG/PNG), returns extracted text |
| `POST` | `/ocr/base64` | Send base64-encoded image string, returns extracted text |

---

## Cost Reference (Lightning.ai)

| GPU | $/hr | Hours from $15 |
|---|---|---|
| T4 | $0.68 | ~22 hrs |
| L4 | $0.70 | ~21 hrs |
| A10G | $1.80 | ~8 hrs |

Sleep the Studio when not in use:
Studio UI → **Stop** button, or call `POST /gpu/sleep?account_id=N` from the gateway.
