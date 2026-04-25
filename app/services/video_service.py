import base64
import logging
import os
import tempfile
import time

import numpy as np
import torch
from PIL import Image

from ..schemas.video import VideoRequest, VideoResponse

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, t2v_pipe, i2v_pipe) -> None:
        self._t2v_pipe = t2v_pipe
        self._i2v_pipe = i2v_pipe
        logger.info(
            "VideoService initialised — T2V: %s | I2V: %s",
            type(t2v_pipe).__name__, type(i2v_pipe).__name__,
        )

    def generate(self, request: VideoRequest) -> VideoResponse:
        clip_count = len(request.prompts)
        logger.info(
            "Video generation started — clips: %d | frames/clip: %d | steps: %d | size: %dx%d | fps: %d",
            clip_count, request.num_frames, request.num_inference_steps,
            request.width, request.height, request.fps,
        )
        t_start = time.perf_counter()

        all_frames = []
        last_frame = None  # PIL Image — last frame of previous clip, seed for I2V

        for idx, prompt in enumerate(request.prompts):
            clip_num = idx + 1
            logger.info(
                "Clip %d/%d — %s — prompt: %.60s...",
                clip_num, clip_count,
                "T2V (first clip)" if idx == 0 else "I2V (continuation)",
                prompt,
            )
            t_clip = time.perf_counter()

            with torch.inference_mode():
                if idx == 0:
                    output = self._t2v_pipe(
                        prompt=prompt,
                        negative_prompt=request.negative_prompt or None,
                        num_frames=request.num_frames,
                        guidance_scale=request.guidance_scale,
                        num_inference_steps=request.num_inference_steps,
                        width=request.width,
                        height=request.height,
                    )
                else:
                    output = self._i2v_pipe(
                        image=last_frame,
                        prompt=prompt,
                        negative_prompt=request.negative_prompt or None,
                        num_frames=request.num_frames,
                        guidance_scale=request.guidance_scale,
                        num_inference_steps=request.num_inference_steps,
                        width=request.width,
                        height=request.height,
                    )

            clip_frames = output.frames[0]  # list[PIL.Image] or list[np.ndarray]
            # I2V pipeline requires PIL Image — convert if diffusers returned ndarray
            last_frame = clip_frames[-1]
            if not isinstance(last_frame, Image.Image):
                last_frame = Image.fromarray(np.uint8(last_frame))
            all_frames.extend(clip_frames)

            logger.info(
                "Clip %d/%d done — %.2fs | %d frames | total so far: %d frames",
                clip_num, clip_count,
                time.perf_counter() - t_clip,
                len(clip_frames), len(all_frames),
            )

        gen_elapsed = time.perf_counter() - t_start
        logger.info(
            "All %d clips generated — %.2fs | %d total frames",
            clip_count, gen_elapsed, len(all_frames),
        )

        logger.info("Encoding %d frames to MP4 ...", len(all_frames))
        t_enc = time.perf_counter()
        video_bytes = self._frames_to_mp4(all_frames, request.fps)
        logger.info(
            "MP4 encoding done — %.2fs | size: %.1f KB",
            time.perf_counter() - t_enc, len(video_bytes) / 1024,
        )

        video_base64 = base64.b64encode(video_bytes).decode()
        total_elapsed = time.perf_counter() - t_start
        duration = len(all_frames) / request.fps

        logger.info(
            "Video generation complete — total: %.2fs | duration: %.1fs | clips: %d | size: %.1f KB",
            total_elapsed, duration, clip_count, len(video_base64) / 1024,
        )

        return VideoResponse(
            video_base64=video_base64,
            num_frames=len(all_frames),
            fps=request.fps,
            duration_seconds=round(duration, 2),
            clips_generated=clip_count,
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
