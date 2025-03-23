import logging
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternVLChatProcessor, InternLM2Tokenizer


from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


import math
from datetime import timedelta

from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs


# The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.
def split_model(model_name, num_layers=None):
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        num_layers = {
            "InternVL2_5-1B": 24,
            "InternVL2_5-2B": 24,
            "InternVL2_5-4B": 36,
            "InternVL2_5-8B": 32,
            "InternVL2_5-26B": 48,
            "InternVL2_5-38B": 64,
            "InternVL2_5-78B": 80,
            "InternVL2-1B": 24,
            "InternVL2-2B": 24,
            "InternVL2-4B": 32,
            "InternVL2-8B": 32,
            "InternVL2-26B": 48,
            "InternVL2-40B": 60,
            "InternVL2-Llama3-76B": 80,
        }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map

PER_IMAGE_NUM_TOKENS = 263 # 258 + 5

@register_model("internvl2")
class InternVL2(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2-2B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        max_num_patches: int = 12,
        max_frame_num_patches: int = 1,
        num_layers=None,
        enable_shared_cross_attention=False,
        enable_cross_attention=False,
        local_attention_group_size=8,
        top_k=100,
        predict_type='key_norms_small',
        adaptive_local_attention=False,
        top_k_starting_layer=0,
        prune_during_prefill_layer_idx=-1,
        prune_for_query=False,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.num_frame = num_frame if isinstance(num_frame, int) else eval(num_frame)
        self.max_num_patches = max_num_patches if isinstance(max_num_patches, int) else eval(max_num_patches)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            device_map = split_model(pretrained.split("/")[-1], num_layers=num_layers)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        local_attention_group_size = eval(local_attention_group_size) if isinstance(local_attention_group_size, str) else local_attention_group_size
        local_attention_group_size = int(local_attention_group_size)
        if local_attention_group_size > 0:
            local_attention_group_size = PER_IMAGE_NUM_TOKENS * local_attention_group_size
        config = InternVLChatConfig.from_pretrained(pretrained, 
            enable_shared_cross_attention=enable_shared_cross_attention, enable_cross_attention=enable_cross_attention, 
            local_attention_group_size=local_attention_group_size, adaptive_local_attention=adaptive_local_attention,
            prune_for_query=prune_for_query)
        config.llm_config.enable_cross_attention = config.enable_cross_attention
        config.llm_config.local_attention_group_size = config.local_attention_group_size
        config.llm_config.enable_shared_cross_attention = config.enable_shared_cross_attention
        config.llm_config.adaptive_local_attention = config.adaptive_local_attention
        config.llm_config.prune_for_query = config.prune_for_query
        self._model = InternVLChatModel.from_pretrained(pretrained, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, device_map=self.device_map).eval()
        self._tokenizer = InternLM2Tokenizer.from_pretrained(self.path, trust_remote_code=True, device_map=device_map)
        for i, decoder_layer in enumerate(self._model.language_model.model.layers):
            if i >= top_k_starting_layer:
                decoder_layer.attention.top_k = top_k
            else:
                decoder_layer.attention.top_k = -1
            decoder_layer.attention.predict_type = predict_type
            if i == prune_during_prefill_layer_idx:
                decoder_layer.prune_during_prefill = True

        self.processor = InternVLChatProcessor(
            self._tokenizer, enable_cross_attention=self._model.config.enable_cross_attention, video_num_segments=self.num_frame, \
            max_num_patches=self.max_num_patches, max_frame_num_patches=max_frame_num_patches)
        
        self.model.img_context_token_id = self.processor.img_context_token_id
        self.model.img_start_token_id = self.processor.img_start_token_id
        self.model.img_end_token_id = self.processor.img_end_token_id
        self.model.bos_token_id = self.processor.bos_token_id


        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            from mantis.models.conversation import conv_templates


            conv = conv_templates['internvl2_5'].copy()
            
            if self.modality == "image":
                if visuals:
                    # visuals = [load_image(visual).to(torch.bfloat16).cuda() for visual in visuals]
                    # pixel_values = torch.cat(visuals, dim=0)
                    # num_patches_list = [visual.size(0) for visual in visuals]
                    num_image_tokens_in_context = contexts.count("<image>")
                    if num_image_tokens_in_context < len(visuals):
                        image_tokens = ["<image>"] * (len(visuals) - num_image_tokens_in_context)
                        image_tokens = " ".join(image_tokens)
                        contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None
                    visuals = None
                conv.append_message(conv.roles[0], contexts)
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()
                num_images_tokens = query.count("<image>")
                query = " ".join(["<image>"] * num_images_tokens) + "\n" + query.replace("<image> ", " ").replace("<image>", "")
                print("Query:", query)
                model_inputs = self.processor(query, images=visuals)
                model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
                for key in model_inputs:
                    if isinstance(model_inputs[key], torch.Tensor):
                        model_inputs[key] = model_inputs[key].to(self.model.device)
                eos_token_id = self.tokenizer.convert_tokens_to_ids(conv.sep.strip())
                generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=eos_token_id)
                responses = self.model.generate(**model_inputs, **generation_config)
                response = self.processor.decode(responses[0], skip_special_tokens=True)
                
            elif self.modality == "video":
                import time
                start = time.time()
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                video_tokens = ["<video>"]
                # contexts = video_tokens + "\n" + contexts
                conv.append_message(conv.roles[0], contexts)
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()
                query = " ".join(video_tokens) + "\n" + query
                model_inputs = self.processor(query, videos=visuals)
                end = time.time()
                print("Video Preprocessing Time:", end - start)
                start = time.time()
                model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
                for key in model_inputs:
                    if isinstance(model_inputs[key], torch.Tensor):
                        model_inputs[key] = model_inputs[key].to(self.model.device)
                eos_token_id = self.tokenizer.convert_tokens_to_ids(conv.sep.strip())
                generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=eos_token_id)
                responses = self.model.generate(**model_inputs, **generation_config)
                response = self.processor.decode(responses[0])
                end = time.time()
                print("Model inference Time:", end - start)
            response = response.strip()
            print("Contexts:", contexts)
            print("Response:", response)
            # exit(1)
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InternVL2")