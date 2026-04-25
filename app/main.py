import logging
import os
from contextlib import asynccontextmanager

import easyocr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

# Must be set before any huggingface_hub / diffusers / transformers import.
# Defaults to a persistent path inside the studio workspace so models are
# cached across restarts and never re-downloaded.
_HF_CACHE = os.environ.get("HF_HOME", "/teamspace/studios/this_studio/hf_cache")
os.environ["HF_HOME"] = _HF_CACHE
os.makedirs(_HF_CACHE, exist_ok=True)

from .routes.ocr import router as ocr_router
from .routes.video import router as video_router
from .services.ocr_service import OCRService
from .services.video_service import VideoService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.info("HuggingFace cache directory: %s", _HF_CACHE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_service = None
    app.state.video_service = None

    logger.info("Loading EasyOCR model (gpu=%s, languages=%s) ...", settings.gpu, settings.languages)
    reader = easyocr.Reader(settings.languages, gpu=settings.gpu)
    app.state.ocr_service = OCRService(reader)
    logger.info("EasyOCR model ready.")

    if settings.enable_video:
        import torch
        from diffusers import WanImageToVideoPipeline, WanPipeline

        device = "cuda" if settings.gpu else "cpu"
        token = settings.hf_token or None

        logger.info("Loading Wan2.2 T2V pipeline (%s) ...", settings.video_model_id)
        t2v_pipe = WanPipeline.from_pretrained(
            settings.video_model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        t2v_pipe.to(device)
        logger.info("Wan2.2 T2V ready.")

        logger.info("Loading Wan2.2 I2V pipeline (%s) ...", settings.video_i2v_model_id)
        i2v_pipe = WanImageToVideoPipeline.from_pretrained(
            settings.video_i2v_model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        i2v_pipe.to(device)
        logger.info("Wan2.2 I2V ready.")

        app.state.video_service = VideoService(t2v_pipe, i2v_pipe)
        logger.info("Both Wan2.2 pipelines loaded — T2V + I2V fully in VRAM (~56 GB).")
    else:
        logger.info("Video generation disabled (ENABLE_VIDEO=false). Skipping Wan2.2 load.")

    yield

    app.state.ocr_service = None
    app.state.video_service = None
    logger.info("Model service shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Model Service",
        description="EasyOCR + Wan2.2 text-to-video on Lightning.ai GPU studio.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ocr_router)
    app.include_router(video_router)

    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status": "ready",
            "ocr": app.state.ocr_service is not None,
            "video": app.state.video_service is not None,
        }

    return app


app = create_app()
