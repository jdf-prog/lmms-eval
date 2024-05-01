import os

AVAILABLE_MODELS = {
    "llava": "Llava",
    "qwen_vl": "Qwen_VL",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "mllava": "MLlava",
    "llava_hf": "LlavaHf",
    "yi_vl": "Yi_VL",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except Exception as e:
        print(f"Error while importing {model_class}: {e}")
        raise e


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
