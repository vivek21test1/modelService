import logging
from contextlib import asynccontextmanager

import easyocr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes.ocr import router as ocr_router
from .routes.video import router as video_router
from .services.ocr_service import OCRService
from .services.video_service import VideoService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_service = None
    app.state.video_service = None

    logger.info("Loading EasyOCR model (gpu=%s, languages=%s) ...", settings.gpu, settings.languages)
    reader = easyocr.Reader(settings.languages, gpu=settings.gpu)
    app.state.ocr_service = OCRService(reader)
    logger.info("EasyOCR model ready.")

    if settings.enable_video:
        logger.info("Loading Wan2.2 pipeline (model_id=%s) ...", settings.video_model_id)
        import torch
        from diffusers import WanPipeline

        pipe = WanPipeline.from_pretrained(
            settings.video_model_id,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda" if settings.gpu else "cpu")
        pipe.enable_model_cpu_offload()
        app.state.video_service = VideoService(pipe)
        logger.info("Wan2.2 pipeline ready.")
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
