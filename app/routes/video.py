import logging
import time

from fastapi import APIRouter, Depends

from ..dependencies import get_video_service
from ..schemas.video import VideoRequest, VideoResponse
from ..services.video_service import VideoService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["Video Generation"])


@router.post("", response_model=VideoResponse)
def generate_video(
    request: VideoRequest,
    service: VideoService = Depends(get_video_service),
) -> VideoResponse:
    logger.info(
        "POST /video — clips: %d | frames: %d | steps: %d",
        len(request.prompts), request.num_frames, request.num_inference_steps,
    )
    t_start = time.perf_counter()
    response = service.generate(request)
    logger.info("POST /video complete — %.2fs", time.perf_counter() - t_start)
    return response
