from __future__ import annotations

import os
import json
import time
import base64
import openai
import requests
from dotenv import load_dotenv
from litestar.params import Body
from litestar.openapi.spec import tag
from typing import Any, Dict, Annotated
from litestar.openapi import OpenAPIConfig
from litestar import Controller, Litestar, get, post
from litestar.config.cors import CORSConfig
from loguru import logger
from utils import stable_base_json, SamplerSet, call_txt2img, LocalLLMDatabaseManager
from payloads import (
    OpenaiChatRequest,
    OllamaChatRequest,
    ImagineRequest,
    GenericResponse,
)

load_dotenv(override=True)
client = openai.OpenAI()

db = LocalLLMDatabaseManager(
    os.getenv("LOCAL_SQL_SERVER"), os.getenv("LOCAL_SQL_DATABASE")
)


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
        logger.info(f"/chat/openai called with: {data}")
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
                data=json.loads(completion.choices[0].model_dump_json()),
            )
        except Exception as e:
            return GenericResponse(success=False, data={"error_message": str(e)})

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
        logger.info(f"chat//ollama called with: {data}")
        try:
            request_payload = {
                "model": data.model if data.model else "Marcus",
                "prompt": data.prompt,
                "stream": False,
            }
            if data.context:
                request_payload["context"] = data.context
            response = requests.post(
                r"http://host.docker.internal:11434/api/generate",
                data=json.dumps(request_payload),
                headers={"Content-Type": "application/json"},
            )
            return GenericResponse(success=True, data=response.json())
        except Exception as e:
            logger.info(e)
            return GenericResponse(
                success=False,
                data={"error_message": str(e)},
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
        """Route to call locally hosted stable diffusion API"""
        logger.info(f"/stable/image called with: {data}")
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
        txt2img_request_payload["sampler_name"] = (
            data.sampler.value if data.sampler else SamplerSet.ddim
        )
        txt2img_request_payload["n_iter"] = data.images.value

        try:
            response = await call_txt2img(payload=txt2img_request_payload)
            if db.active:
                for image in response["images"]:
                    try:
                        s = time.time()
                        logger.info("converting base64 image data to binary.")
                        image_binary = base64.b64decode(image)
                        logger.info("attempting to insert image request into db")
                        db_response = db.insert_stable_request(
                            user_message=data.prompt,
                            negative_prompt=data.negative_prompt,
                            image=image_binary,
                        )
                        logger.info(f"db response: {db_response}")
                        end = time.time() - s
                        logger.info(f"time elapsed inserting image into db: {end}")
                    except Exception as db_error:
                        logger.info(f"db insert failed: {db_error}")
        except Exception as e:
            return GenericResponse(
                success=False,
                response={"data": {"message": f"An error has occured: {e}"}},
            )

        return GenericResponse(success=True, response=response)


app = Litestar(
    route_handlers=[ChatController, StableController],
    openapi_config=OpenAPIConfig(title="Local LLM API", version="1.0.0"),
    allowed_hosts=["*"],
    cors_config=CORSConfig(allow_origins=["*"]),
)
