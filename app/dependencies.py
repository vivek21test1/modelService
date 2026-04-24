from fastapi import HTTPException, Request

from .services.ocr_service import OCRService
from .services.video_service import VideoService


def get_ocr_service(request: Request) -> OCRService:
    service: OCRService | None = request.app.state.ocr_service
    if service is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    return service


def get_video_service(request: Request) -> VideoService:
    service: VideoService | None = request.app.state.video_service
    if service is None:
        raise HTTPException(status_code=503, detail="Video model not available. Set ENABLE_VIDEO=true and restart.")
    return service
