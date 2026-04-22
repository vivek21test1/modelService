import logging
from contextlib import asynccontextmanager

import easyocr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes.ocr import router as ocr_router
from .services.ocr_service import OCRService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_service = None
    logger.info("Loading EasyOCR model (gpu=%s, languages=%s) ...", settings.gpu, settings.languages)
    reader = easyocr.Reader(settings.languages, gpu=settings.gpu)
    app.state.ocr_service = OCRService(reader)
    logger.info("EasyOCR model ready.")
    yield
    app.state.ocr_service = None
    logger.info("Model service shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EasyOCR Model Service",
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
    return app


app = create_app()
