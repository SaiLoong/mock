# -*- coding: utf-8 -*-
# @file aw_generate_image_by_metadata.py
# @author zhangshilong
# @date 2024/9/4

import json
import math
import os
import random

import jsonlines
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from diffusers.utils import pt_to_pil

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”


def read_json(path, *args, **kwargs):
    with open(path, "r") as file:
        return json.load(file, *args, **kwargs)


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


def plot_images(images, n_rows=None, n_cols=None, suptitle=None, titles=None, scale=1.):
    if isinstance(images, Image.Image):
        images = [images]
    elif isinstance(images, torch.Tensor):
        images = pt_to_pil(images)

    n = len(images)
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n / n_rows)
    elif n_rows is None:
        n_rows = math.ceil(n / n_cols)

    titles = titles or [None] * n
    assert len(titles) == n, f"titles({len(titles)})应和images({n})长度一致"

    width, height = images[0].size
    dpi = 96
    width_figsize = int(n_cols * width / dpi * scale)
    height_figsize = int(n_rows * height / dpi * scale)

    # 等价于 fig = plt.figure(figsize=..., layout=...)    axes = fig.subplots(n_rows, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_figsize, height_figsize), layout="constrained")
    fig.suptitle(suptitle)
    axes = [axes] if isinstance(axes, Axes) else axes.flatten()

    # axes长度可能会大于images
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)

    # 空白图片也有坐标轴，也需要取消
    for ax in axes:
        ax.axis("off")
    plt.show()


def plot_image_with_prompt(image, prompt):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(prompt, "\n\n")


# ================================================================================================================


model_path = "/mnt/workspace/model/awportrait_v14"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")

# DPM++ 2M Karras
assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
assert pipe.scheduler.config["use_karras_sigmas"] is True


def generate(prompt):
    negative_prompt = "ng_deepnegative_v1_75t,(badhandv4:1.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,((monochrome)),((grayscale)) watermark,moles,large breast,big breast,long fingers:1 bad hand:1,many legs,many shoes,"
    generator = torch.Generator("cuda").manual_seed(1024)

    image = pipe(
        prompt, negative_prompt=negative_prompt, generator=generator,
        num_inference_steps=40, guidance_scale=7,
        height=768, width=512
    ).images[0]
    return image


# ================================================================================================================


image_dir = "/mnt/workspace/dataset/yuer/train"
metadata_path = os.path.join(image_dir, "metadata.jsonl")
metadata = read_jsonl(metadata_path)

generate_dir = "/mnt/workspace/output/yuer_awportrait_v14_generate"
os.makedirs(generate_dir, exist_ok=True)

# ================================================================================================================


# 抽样
for meta in random.sample(metadata, k=5):
    filename = meta["file_name"]
    prompt = meta["text"]
    image = generate(prompt)

    plot_image_with_prompt(image, prompt)

    generate_path = os.path.join(generate_dir, filename)
    image.save(generate_path, quality=95, subsampling=0)

# ================================================================================================================


# 全量
pipe.set_progress_bar_config(disable=True)
meta_dict = dict()
# 3.15s/it
for meta in tqdm(metadata):
    filename = meta["file_name"]
    prompt = meta["text"]

    generate_path = os.path.join(generate_dir, filename)
    if os.path.isfile(generate_path):
        image = Image.open(generate_path)
    else:
        image = generate(prompt)
        image.save(generate_path, quality=95, subsampling=0)

    meta_dict[filename] = (prompt, image)

# ================================================================================================================


# 抽样展示
N = 10
reserved_filenames = ["8330_17.jpg", "8315_09.jpg", "7557_27.jpg"]
candidate_filenames = set(meta_dict.keys()).difference(reserved_filenames)
select_filenames = reserved_filenames + random.sample(candidate_filenames, k=N - len(reserved_filenames))

select_prompts, select_images = zip(*[meta_dict[filename] for filename in select_filenames])

plot_images(select_images, n_cols=5, suptitle=f"AWportrait_v14生成结果抽样", titles=select_filenames, scale=0.6)

for idx, (filename, prompt) in enumerate(zip(select_filenames, select_prompts), start=1):
    print(f"[{idx}]\n{filename}\n{prompt}\n")
