from pydantic import BaseModel as _BaseModel

class BaseModel(_BaseModel):
    """Extend Pydantic's BaseModel to enable ORM mode"""

    model_config = {"from_attributes": True}

from utils import (
    Upscale,
    UpscaleModel,
    SamplerSet,
    Images
)


class OllamaChatRequest(BaseModel):
    prompt: str
    model: str | None


class OpenaiChatRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    model: str
    max_tokens: int


class GenericResponse(BaseModel):
    success: bool
    response: dict | None
    

class ImagineRequest(BaseModel):
    prompt: str
    negative_prompt: str
    quality: Upscale
    cfg_scale: float
    steps: int
    seed: int
    upscale_model: UpscaleModel = UpscaleModel.latent
    sampler: SamplerSet = SamplerSet.ddim
    images: Images = Images.two
