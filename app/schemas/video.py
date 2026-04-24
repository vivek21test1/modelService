from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the video to generate")
    negative_prompt: str = Field("", description="What to avoid in the video")
    num_frames: int = Field(81, ge=16, le=121, description="Number of frames (81 = ~5s at 16fps)")
    fps: int = Field(16, ge=8, le=30, description="Frames per second for the output video")
    guidance_scale: float = Field(5.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=100)
    width: int = Field(832, description="Must be divisible by 32")
    height: int = Field(480, description="Must be divisible by 32")


class VideoResponse(BaseModel):
    video_base64: str = Field(..., description="Base64-encoded MP4 video")
    num_frames: int
    fps: int
    duration_seconds: float
