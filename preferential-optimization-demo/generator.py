from __future__ import annotations

import os
import requests
import tempfile
import time
from typing import NoReturn

import base64
from io import BytesIO
from PIL import Image

from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna_dashboard import register_preference_feedback_component
from optuna_dashboard.preferential import create_study
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler


STORAGE_URL = "sqlite:///example.db"
artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
artifact_store = FileSystemArtifactStore(base_path=artifact_path)
os.makedirs(artifact_path, exist_ok=True)


def gen_image(params):
    loras = " ".join([
        "<lora:Concept Art Twilight Style SDXL_LoRA_Pony Diffusion V6 XL:{twilight_weight}>"
        "<lora:add-detail-xl:{detail_weight}>",
        "<lora:Vintage_Street_Photo:{vintage_weight}>",
        "<lora:xl_more_art-full_v1:{art_weight}>"
    ]).format(**params)
    positive = "score_9, score_8_up, score_7_up, an extremely beautiful colorful sunset, valley, sky, natural" + loras
    negative = "score_6, score_5, score_4, monochrome"
    data_template = {
        "alwayson_scripts": {
            "ADetailer": {
            "args": [
                False,
                False,
                {
                "ad_cfg_scale": 7,
                "ad_checkpoint": "Use same checkpoint",
                "ad_clip_skip": 1,
                "ad_confidence": 0.3,
                "ad_controlnet_guidance_end": 1,
                "ad_controlnet_guidance_start": 0,
                "ad_controlnet_model": "None",
                "ad_controlnet_module": "None",
                "ad_controlnet_weight": 1,
                "ad_denoising_strength": 0.4,
                "ad_dilate_erode": 4,
                "ad_inpaint_height": 512,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_inpaint_width": 512,
                "ad_mask_blur": 4,
                "ad_mask_k_largest": 0,
                "ad_mask_max_ratio": 1,
                "ad_mask_merge_invert": "None",
                "ad_mask_min_ratio": 0,
                "ad_model": "face_yolov8n.pt",
                "ad_model_classes": "",
                "ad_negative_prompt": "",
                "ad_noise_multiplier": 1,
                "ad_prompt": "",
                "ad_restore_face": False,
                "ad_sampler": "DPM++ 2M Karras",
                "ad_scheduler": "Use same scheduler",
                "ad_steps": 28,
                "ad_use_cfg_scale": False,
                "ad_use_checkpoint": False,
                "ad_use_clip_skip": False,
                "ad_use_inpaint_width_height": False,
                "ad_use_noise_multiplier": False,
                "ad_use_sampler": False,
                "ad_use_steps": False,
                "ad_use_vae": False,
                "ad_vae": "Use same VAE",
                "ad_x_offset": 0,
                "ad_y_offset": 0,
                "is_api": []
                },
                {
                "ad_cfg_scale": 7,
                "ad_checkpoint": "Use same checkpoint",
                "ad_clip_skip": 1,
                "ad_confidence": 0.3,
                "ad_controlnet_guidance_end": 1,
                "ad_controlnet_guidance_start": 0,
                "ad_controlnet_model": "None",
                "ad_controlnet_module": "None",
                "ad_controlnet_weight": 1,
                "ad_denoising_strength": 0.4,
                "ad_dilate_erode": 4,
                "ad_inpaint_height": 512,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_inpaint_width": 512,
                "ad_mask_blur": 4,
                "ad_mask_k_largest": 0,
                "ad_mask_max_ratio": 1,
                "ad_mask_merge_invert": "None",
                "ad_mask_min_ratio": 0,
                "ad_model": "None",
                "ad_model_classes": "",
                "ad_negative_prompt": "",
                "ad_noise_multiplier": 1,
                "ad_prompt": "",
                "ad_restore_face": False,
                "ad_sampler": "DPM++ 2M Karras",
                "ad_scheduler": "Use same scheduler",
                "ad_steps": 28,
                "ad_use_cfg_scale": False,
                "ad_use_checkpoint": False,
                "ad_use_clip_skip": False,
                "ad_use_inpaint_width_height": False,
                "ad_use_noise_multiplier": False,
                "ad_use_sampler": False,
                "ad_use_steps": False,
                "ad_use_vae": False,
                "ad_vae": "Use same VAE",
                "ad_x_offset": 0,
                "ad_y_offset": 0,
                "is_api": []
                }
            ]
            },
            "API payload": {
            "args": []
            },
            "Dynamic Prompts v2.17.1": {
            "args": [
                False,
                True,
                1,
                False,
                False,
                False,
                1.1,
                1.5,
                100,
                0.7,
                True,
                False,
                True,
                False,
                False,
                0,
                "Gustavosta/MagicPrompt-Stable-Diffusion",
                ""
            ]
            },
            "Extra options": {
            "args": []
            },
            "Hypertile": {
            "args": []
            },
            "Refiner": {
            "args": [
                False,
                "",
                0.8
            ]
            },
            "Seed": {
            "args": [
                -1,
                False,
                -1,
                0,
                0,
                0
            ]
            }
        },
        "batch_size": 1,
        "cfg_scale": 7,
        "comments": {},
        "disable_extra_networks": False,
        "do_not_save_grid": False,
        "do_not_save_samples": False,
        "enable_hr": False,
        "height": 1216,
        "hr_negative_prompt": "",
        "hr_prompt": "",
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_scale": 1.5,
        "hr_second_pass_steps": 10,
        "hr_upscaler": "Latent",
        "n_iter": 1,
        "negative_prompt": negative,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "prompt": positive,
        "restore_faces": False,
        "s_churn": 0,
        "s_min_uncond": 0,
        "s_noise": 1,
        "s_tmax": None,
        "s_tmin": 0,
        "sampler_name": "DPM++ 2M Karras",
        "script_args": [],
        "script_name": None,
        "seed": -1,
        "seed_enable_extras": True,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 30,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 832
    }
    url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=data)
    # print(response.__dict__)
    response_data = response.json()
    # print(response_data)

    images = response_data.get('images', [])
    decoded_images = []
    
    for b64_image in images:
        image_data = base64.b64decode(b64_image)
        image = Image.open(BytesIO(image_data))
        decoded_images.append(image)
    
    return decoded_images


def main() -> NoReturn:
    study = create_study(
        n_generate=4,
        study_name="Preferential Optimization",
        storage=STORAGE_URL,
        sampler=PreferentialGPSampler(),
        load_if_exists=True,
    )
    # Change the component, displayed on the human feedback pages.
    # By default (component_type="note"), the Trial's Markdown note is displayed.
    user_attr_key = "rgb_image"
    register_preference_feedback_component(study, "artifact", user_attr_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            # If study.should_generate() returns False,
            # the generator waits for human evaluation.
            if not study.should_generate():
                time.sleep(0.1)  # Avoid busy-loop
                continue

            trial = study.ask()
            # 1. Ask new parameters
            r = trial.suggest_int("r", 0, 255)
            g = trial.suggest_int("g", 0, 255)
            b = trial.suggest_int("b", 0, 255)

            # 2. Generate image
            image_path = os.path.join(tmpdir, f"sample-{trial.number}.png")
            image = Image.new("RGB", (320, 240), color=(r, g, b))
            image.save(image_path)

            # 3. Upload Artifact and set artifact_id to trial.user_attrs["rgb_image"].
            artifact_id = upload_artifact(trial, image_path, artifact_store)
            trial.set_user_attr(user_attr_key, artifact_id)


if __name__ == "__main__":
    main()
