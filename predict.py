# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda")

img_path = "dataset/UniMER-Test/cpe/0000013.png"
img = Image.open(img_path).convert("RGB")
img_stem = Path(img_path).stem

s1 = time.perf_counter()
print("Loading model")
model_path = "outputs/checkpoint-27738"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
print("Finished loading model.")

s2 = time.perf_counter()

pixel_values = processor(img, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)
print(pixel_values.shape)

s3 = time.perf_counter()

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
print(f"loading_model: {s2 - s1}s")
print(f"infer: {s3 - s2}s")
