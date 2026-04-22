import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..dependencies import get_ocr_service
from ..schemas.ocr import HealthResponse, OCRBase64Request, OCRResponse
from ..services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OCR"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    loaded = request.app.state.ocr_service is not None
    status = "ready" if loaded else "loading"
    logger.debug("Health check — model status: %s", status)
    return HealthResponse(status=status, model_loaded=loaded, gpu=True)


@router.post("/ocr", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service),
):
    logger.info("POST /ocr — filename: %s | content-type: %s", file.filename, file.content_type)
    t_start = time.perf_counter()

    image_bytes = await file.read()
    result = service.process_bytes(image_bytes)

    logger.info("POST /ocr — completed in %.2fs", time.perf_counter() - t_start)
    return result


@router.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(
    body: OCRBase64Request,
    service: OCRService = Depends(get_ocr_service),
):
    logger.info("POST /ocr/base64 — languages: %s", body.languages)
    t_start = time.perf_counter()

    try:
        result = service.process_base64(body.image_base64)
    except Exception:
        logger.warning("POST /ocr/base64 — failed to decode base64 payload")
        raise HTTPException(status_code=400, detail="Invalid base64 image string.")

    logger.info("POST /ocr/base64 — completed in %.2fs", time.perf_counter() - t_start)
    return result
