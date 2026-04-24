from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    gpu: bool = True
    languages: list[str] = ["en"]

    # Wan2.2 text-to-video — disabled by default to save GPU memory when not needed
    enable_video: bool = True
    video_model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # HuggingFace token — required for gated models like Wan2.2
    hf_token: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
