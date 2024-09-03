# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/8/31

import torch

from diffusers import StableDiffusionPipeline

model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               torch_dtype=torch.float16, device_map="balanced")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
