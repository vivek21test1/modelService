from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="One prompt per clip. Clip 1 uses T2V, clips 2-N continue from the previous clip's last frame via I2V.",
    )
    negative_prompt: str = Field("", description="Applied to all clips")
    num_frames: int = Field(121, ge=16, le=121, description="Frames per clip (121 = ~6s at 20fps)")
    fps: int = Field(20, ge=8, le=30, description="Frames per second for the output video")
    guidance_scale: float = Field(6.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(35, ge=10, le=100)
    width: int = Field(480, description="Must be divisible by 32")
    height: int = Field(832, description="Must be divisible by 32")


class VideoResponse(BaseModel):
    video_base64: str = Field(..., description="Base64-encoded MP4 video")
    num_frames: int
    fps: int
    duration_seconds: float
    clips_generated: int
