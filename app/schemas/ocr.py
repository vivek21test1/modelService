from pydantic import BaseModel


class OCRBase64Request(BaseModel):
    image_base64: str
    languages: list[str] = ["en"]


class OCRResponse(BaseModel):
    text: str
    lines: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu: bool
