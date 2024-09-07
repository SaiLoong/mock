# -*- coding: utf-8 -*-
# @file generate_metadata.py
# @author zhangshilong
# @date 2024/9/7

import json
import os
import random

import jsonlines
import matplotlib.pyplot as plt
from PIL import Image

from transformers import CLIPTokenizer


def read_json(path, *args, **kwargs):
    with open(path, "r") as file:
        return json.load(file, *args, **kwargs)


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


def write_jsonl(path, data):
    with jsonlines.open(path, "w") as f:
        f.write_all(data)


# =======================================================================================


# 抽查raw_caption有没问题
image_dir = "/mnt/workspace/dataset/yuer/train"
filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
filenames.sort()

raw_caption_path = os.path.join(image_dir, "raw_caption.json")
raw_caption = read_json(raw_caption_path)

for filename in random.sample(filenames, k=5):
    image = Image.open(os.path.join(image_dir, filename))
    caption = raw_caption[filename]

    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(caption, "\n\n")

# =======================================================================================


# 后处理生成metadata
sd_model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
clip_tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")

insert_tags = ["Yuer", "1girl", "solo", "portrait", "20-year-old"]
n = len(insert_tags)

metadata = list()
for filename in filenames:
    caption = raw_caption[filename]
    tags = [tag.strip() for tag in caption.split(",")[:-1]]  # 最后的tag可能不完整

    # 将固定tag随机插到前10的位置
    for idx, tag in enumerate(insert_tags):
        if tag in tags:
            continue

        index = random.randrange(10 - n + idx)
        tags.insert(index, tag)

    caption = ", ".join(tags)
    assert len(clip_tokenizer(caption).input_ids) <= clip_tokenizer.model_max_length
    metadata.append({"file_name": filename, "text": caption})

metadata_path = os.path.join(image_dir, "metadata.jsonl")
write_jsonl(metadata_path, metadata)
