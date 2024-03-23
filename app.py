from __future__ import annotations

import openai

import json
import requests

from asyncio import sleep
from typing import Any, Dict, Annotated
from litestar.openapi import OpenAPIConfig
from litestar import Controller, Litestar, get, post
from litestar.openapi.spec import tag
from litestar.params import Body
from payloads import (
    OpenaiChatRequest,
    OllamaChatRequest,
    ImagineRequest,
    GenericResponse,
)

from utils import stable_base_json, SamplerSet, call_txt2img

from dotenv import load_dotenv

load_dotenv(override=True)

client = openai.OpenAI()


class ChatController(Controller):
    path = "/chat"
    tags = ["Chat"]

    @post(path="/openai")
    async def openai(
        self,
        data: Annotated[
            OpenaiChatRequest,
            Body(
                title="OpenaiChatRequest",
                description="Request JSON to ping the /chat/openai route.",
            ),
        ],
    ) -> GenericResponse:
        """Route to ping and chat with Openai"""
        try:
            completion = client.chat.completions.create(
                model=data.model,
                messages=[
                    {
                        "role": "system",
                        "content": data.system_prompt,
                    },
                    {"role": "user", "content": data.user_prompt},
                ],
                max_tokens=data.max_tokens,
            )

            return GenericResponse(
                success=True,
                response={"response": completion.choices[0].message.content},
            )
        except Exception as e:
            return GenericResponse(
                success=False, response={"response": f"An error has occured --> {e}"}
            )

    @post(path="/ollama")
    async def ollama(
        self,
        data: Annotated[
            OllamaChatRequest,
            Body(
                title="OllamaChatRequest",
                description="Request JSON to ping the /chat/ollama route.",
            ),
        ],
    ) -> GenericResponse:
        """Route Handler that interfaces with Local / Non-Local LLM"""

        print(data)

        try:
            request_payload = {
                "model": data.model if data.model else "Marcus",
                "prompt": data.prompt,
                "stream": False,
            }
            response = requests.post(
                r"http://127.0.0.1:11434/api/generate",
                data=json.dumps(request_payload),
                headers={"Content-Type": "application/json"},
            )

            response_json = json.loads(response.content.decode("utf8"))

            return GenericResponse(success=True, response=response_json)
        except Exception as e:
            return GenericResponse(
                success=False, response={"response": {"message": str(e)}}
            )


class StableController(Controller):
    path = "/stable"
    tags = ["Stable"]

    @post("/imagine")
    async def imagine(
        self,
        data: Annotated[
            ImagineRequest,
            Body(
                title="ImagineRequest",
                description="Request JSON to ping the /stable/imagine route.",
            ),
        ],
    ) -> GenericResponse:
        
        txt2img_request_payload = stable_base_json.copy()
        txt2img_request_payload["hr_negative_prompt"] = data.negative_prompt
        txt2img_request_payload["negative_prompt"] = data.negative_prompt
        txt2img_request_payload["hr_prompt"] = data.prompt
        txt2img_request_payload["prompt"] = data.prompt
        txt2img_request_payload["hr_scale"] = data.quality.value
        txt2img_request_payload["cfg_scale"] = data.cfg_scale
        txt2img_request_payload["steps"] = data.steps
        txt2img_request_payload["seed"] = data.seed
        txt2img_request_payload["hr_upscaler"] = data.upscale_model.value
        txt2img_request_payload["sampler_name"] = data.sampler.value if data.sampler else SamplerSet.ddim
        txt2img_request_payload["n_iter"] = data.images.value
        
        try:
            response = await call_txt2img(payload=txt2img_request_payload)
        except Exception as e:
            return GenericResponse(success=False, response={"response": {"message": f"An error has occured: {e}"}})
        
        return GenericResponse(
            success=True, response=response
        )


app = Litestar(
    route_handlers=[ChatController, StableController],
    openapi_config=OpenAPIConfig(title="Local LLM API", version="1.0.0"),
)
