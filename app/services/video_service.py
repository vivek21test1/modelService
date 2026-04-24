import base64
import logging
import os
import tempfile
import time

import numpy as np
import torch

from ..schemas.video import VideoRequest, VideoResponse

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, pipe) -> None:
        self._pipe = pipe
        logger.info("VideoService initialised — model: %s", type(pipe).__name__)

    def generate(self, request: VideoRequest) -> VideoResponse:
        logger.info(
            "Video generation started — prompt: %d chars | frames: %d | steps: %d | size: %dx%d",
            len(request.prompt), request.num_frames,
            request.num_inference_steps, request.width, request.height,
        )
        t_start = time.perf_counter()

        with torch.inference_mode():
            output = self._pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_frames=request.num_frames,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                width=request.width,
                height=request.height,
            )

        frames = output.frames[0]   # list[PIL.Image]
        gen_elapsed = time.perf_counter() - t_start
        logger.info(
            "Wan2.2 inference done — %.2fs | %d frames generated",
            gen_elapsed, len(frames),
        )

        logger.info("Encoding frames to MP4 ...")
        t_enc = time.perf_counter()
        video_bytes = self._frames_to_mp4(frames, request.fps)
        logger.info(
            "MP4 encoding done — %.2fs | size: %.1f KB",
            time.perf_counter() - t_enc, len(video_bytes) / 1024,
        )

        video_base64 = base64.b64encode(video_bytes).decode()
        total_elapsed = time.perf_counter() - t_start
        duration = len(frames) / request.fps

        logger.info(
            "Video generation complete — total: %.2fs | duration: %.1fs | base64: %.1f KB",
            total_elapsed, duration, len(video_base64) / 1024,
        )

        return VideoResponse(
            video_base64=video_base64,
            num_frames=len(frames),
            fps=request.fps,
            duration_seconds=round(duration, 2),
        )

    @staticmethod
    def _frames_to_mp4(frames, fps: int) -> bytes:
        import imageio

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name

            imageio.mimsave(
                tmp_path,
                [np.array(frame) for frame in frames],
                fps=fps,
                codec="libx264",
                quality=8,
            )

            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
