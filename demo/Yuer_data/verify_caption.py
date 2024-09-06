# -*- coding: utf-8 -*-
# @file verify_caption.py
# @author zhangshilong
# @date 2024/9/4

import math
import os

import jsonlines
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.utils import pt_to_pil

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”


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


# ================================================================================================================


model_path = "/mnt/workspace/model/awportrait_v14"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")


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


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


image_dir = "/mnt/workspace/dataset/yuer/train2"
metadata_path = os.path.join(image_dir, "metadata.jsonl")
metadata = read_jsonl(metadata_path)
N = len(metadata)

# ================================================================================================================


# BUG 容易生成娃娃脸
generate_dir = "/mnt/workspace/output/yuer_awportrait_v14_generate"
os.makedirs(generate_dir, exist_ok=True)
images = list()
titles = list()

for meta in tqdm(metadata):
    filename = meta["file_name"]
    prompt = meta["text"]
    title = filename

    prompt = prompt.replace("Yuer, ", "")
    print(f"[{filename}] {prompt}")

    image = generate(prompt)

    images.append(image)
    titles.append(title)

    generate_path = os.path.join(generate_dir, filename)
    image.save(generate_path, quality=95, subsampling=0)

plot_images(images, n_cols=10, suptitle=f"AWportrait_v14生成结果（共{N}张）", titles=titles, scale=0.15)
