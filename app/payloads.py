from pydantic import BaseModel as _BaseModel
from typing import Optional, List
from utils import Upscale, UpscaleModel, SamplerSet, Images


class BaseModel(_BaseModel):
    """Extend Pydantic's BaseModel to enable ORM mode"""

    model_config = {"from_attributes": True}


class OllamaChatRequest(BaseModel):
    prompt: str
    model: str = "Jade"
    context: Optional[List[int]] = None


class OpenaiChatRequest(BaseModel):
    system_prompt: str = "You are an helpful AI assistant."
    user_prompt: str
    model: str = "gpt-4o"
    max_tokens: int = 500


class GenericResponse(BaseModel):
    success: bool
    data: dict = {}


class ImagineRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    quality: Upscale = Upscale.two
    cfg_scale: float = 5.0
    steps: int = 25
    seed: int = -1
    upscale_model: UpscaleModel = UpscaleModel.latent
    sampler: SamplerSet = SamplerSet.ddim
    images: Images = Images.two
