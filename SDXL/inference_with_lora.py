# -*- coding: utf-8 -*-
# @file inference_with_lora.py
# @author zhangshilong
# @date 2024/9/3

import torch

from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline

base_path = "/mnt/workspace/model/stable-diffusion-xl-base-1.0"
vae_path = "/mnt/workspace/model/sdxl-vae-fp16-fix"

# SDXL自带的VAE容易出现数值不稳定，解码时容易变成全黑图
# 详见https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, device_map="auto")
# Pipeline可以直接使用DiffusionPipeline，明确写更方便进入源码
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_path, vae=vae,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)

prompt = "a blue and white dragon with its mouth open"

# 原模型效果
image = pipe(prompt).images[0]

# 加载lora权重和层
lora_path = "/mnt/workspace/LoRA_model/sdxl-pokemon"
pipe.load_lora_weights(lora_path)
# lora效果
image2 = pipe(prompt).images[0]

# ======================================================


# 融合与解除融合
# TODO 这个有点危险，来回操作、fuse_lora叠加不知道把权重搞成什么样的
pipe.fuse_lora(lora_scale=1.0)
pipe.unfuse_lora()

# 彻底卸载lora权重和层，无论之前权重fuse得怎样都能还原
pipe.unload_lora_weights()
