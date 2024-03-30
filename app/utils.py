import os
import json
import pyodbc
import typing
import asyncio
import requests
import functools
from enum import Enum


def to_thread(func: typing.Callable) -> typing.Coroutine:
    """Function to wrap another function in a to_thread
    so it can be awaited and non-blocking.

    Args:
        func (typing.Callable): function to wrap

    Returns:
        typing.Coroutine: awaitable coroutine
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@to_thread
def call_txt2img(payload: dict):
    """Function to call the stable diffusion API.

    Args:
        payload (dict): request to be generated.

    Returns:
        _type_: JSON response from the stable API.
    """
    # call the api
    response = requests.post(
        url=r"http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload
    ).json()
    return response


# upscale options
class Upscale(Enum):
    one = 1
    two = 2
    three = 3


class UpscaleModel(Enum):
    none = "None"
    lanczos = "Lanczos"
    latent = "Latent"
    latent_antialiased = "Latent (antialiased)"
    latent_bicubic = "Latent (bicubic)"
    latent_bicubic_antialiased = "Latent (bicubic antialiased)"
    latent_nearest = "Latent (nearest)"
    latent_nearest_exact = "Latent (nearest-exact)"
    nearest = "Nearest"
    esrgan_4x = "ESRGAN_4x"
    ldsr = "LDSR"
    r_esrgan_4x = "R-ESRGAN 4x+"
    r_esrgan_4x_anime6b = "R-ESRGAN 4x+ Anime6B"
    scunet_gan = "ScuNET GAN"
    scunet_psnr = "ScuNET PSNR"
    swinir_4x = "SwinIR 4x"


class SamplerSet(Enum):
    none = None
    dpm_2m_karras = "DPM++ 2M Karras"
    dpm_sde_karras = "DPM++ SDE Karras"
    dpm_2m_sde_exponential = "DPM++ 2M SDE Exponential"
    dpm_2m_sde_karras = "DPM++ 2M SDE Karras"
    euler_a = "Euler a"
    euler = "Euler"
    lms = "LMS"
    heun = "Heun"
    dpm2 = "DPM2"
    dpm2_a = "DPM2 a"
    dpm_2s_a = "DPM++ 2S a"
    dpm_2m = "DPM++ 2M"
    dpm_sde = "DPM++ SDE"
    dpm_2m_sde = "DPM++ 2M SDE"
    dpm_2m_sde_heun = "DPM++ 2M SDE Heun"
    dpm_2m_sde_heun_karras = "DPM++ 2M SDE Heun Karras"
    dpm_2m_sde_heun_exponential = "DPM++ 2M SDE Heun Exponential"
    dpm_3m_sde = "DPM++ 3M SDE"
    dpm_3m_sde_karras = "DPM++ 3M SDE Karras"
    dpm_3m_sde_exponential = "DPM++ 3M SDE Exponential"
    dpm_fast = "DPM fast"
    dpm_adaptive = "DPM adaptive"
    lms_karras = "LMS Karras"
    dpm2_karras = "DPM2 Karras"
    dpm2_a_karras = "DPM2 a Karras"
    dpm_2s_a_karras = "DPM++ 2S a Karras"
    restart = "Restart"
    ddim = "DDIM"
    plms = "PLMS"
    unipc = "UniPC"


class Images(Enum):
    four = 4
    three = 3
    two = 2
    one = 1


# payload to send to txt2img
stable_base_json = {
    "alwayson_scripts": {
        "API payload": {"args": []},
        "Additional networks for generating": {
            "args": [
                False,
                False,
                "LoRA",
                "None",
                0,
                0,
                "LoRA",
                "None",
                0,
                0,
                "LoRA",
                "None",
                0,
                0,
                "LoRA",
                "None",
                0,
                0,
                "LoRA",
                "None",
                0,
                0,
                None,
                "Refresh models",
            ]
        },
        "Dynamic Prompts v2.17.1": {
            "args": [
                True,
                False,
                1,
                False,
                False,
                False,
                1.1,
                1.5,
                100,
                0.7,
                False,
                False,
                True,
                False,
                False,
                0,
                "Gustavosta/MagicPrompt-Stable-Diffusion",
                "",
            ]
        },
        "Extra options": {"args": []},
        "Hypertile": {"args": []},
        "Refiner": {"args": [False, "", 0.8]},
        "Seed": {"args": [-1, False, -1, 0, 0, 0]},
    },
    "batch_size": 1,
    "batch_count": 4,
    "cfg_scale": 3,
    "comments": {},
    "denoising_strength": 0.7,
    "disable_extra_networks": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "enable_hr": True,
    "height": 480,
    "hr_negative_prompt": "",
    "hr_prompt": "",
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_scale": 3,
    "hr_second_pass_steps": 0,
    "hr_upscaler": "Latent",
    "n_iter": 4,
    "negative_prompt": "",
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "prompt": "",
    "restore_faces": False,
    "s_churn": 0.0,
    "s_min_uncond": 0.0,
    "s_noise": 1.0,
    "s_tmax": None,
    "s_tmin": 0.0,
    "sampler_name": "DDIM",
    "script_args": [],
    "script_name": None,
    "seed": -1,
    "seed_enable_extras": True,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "steps": 20,
    "styles": [],
    "subseed": 2408667576,
    "subseed_strength": 0,
    "tiling": False,
    "width": 800,
}


class LocalLLMDatabaseManager:
    def __init__(self, server: str, database: str):
        self.__server = server
        self.__database = database
        self.active = not (self.__server is None or self.__database is None)
    def ___get_connection(self):
        connection_strig = (
            r"Driver={SQL Server};"
            f"Server={self.__server};"
            f"Database={self.__database};"
            r"Trusted_Connection=yes;"
        )

        return pyodbc.connect(connection_strig)

    def insert_chat_request(
        self,
        user_message: str,
        ai_response: str | None,
        model: str | None,
        json_payload: str | None,
    ) -> str:
        response = None

        conn = self.___get_connection()
        cursor = conn.cursor()

        insert_query = """
set xact_abort on
Begin Try
  Begin Tran
    INSERT INTO [LocalLLMApi].[dbo].[ChatRequests] (DatetimeUTC, UserMessage, AIResponse, Model, JsonPayload) VALUES (GETUTCDATE(), ?, ?, ?, ?)
  Commit Tran
End Try

Begin Catch
   Rollback Tran
End Catch
"""
        try:
            cursor.execute(
                insert_query, (user_message, ai_response, model, json_payload)
            )
            conn.commit()
            response = "Successfully inserted the records into the database."
        except Exception as e:
            response = f"Failed to insert the records into the database: {e}."

        cursor.close()
        conn.close()

        return response

    def insert_stable_request(
        self, user_message: str | None, negative_prompt: str | None, image: str | None
    ) -> str:
        conn = self.___get_connection()
        cursor = conn.cursor()

        response = None

        

        insert_query = """
set xact_abort on
Begin Try
  Begin Tran
    INSERT INTO [LocalLLMApi].[dbo].[StableRequests] (DatetimeUTC, UserMessage, Image, NegativePrompt) VALUES (GETUTCDATE(), ?, ?, ?)
  Commit Tran
End Try

Begin Catch
   Rollback Tran
End Catch
"""
        
        try:
            cursor.execute(insert_query, (user_message, image, negative_prompt))
            conn.commit()
            response = "Successfully inserted image into the database."
        except Exception as e:
            response = f"Failed to insert the image into the database: {e}."

        cursor.close()
        conn.close()

        return response
