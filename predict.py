import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from transformers import (
    PreTrainedTokenizerFast,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [
        "".join(detok.split(" "))
        .replace("Ġ", " ")
        .replace("[EOS]", "")
        .replace("[BOS]", "")
        .replace("[PAD]", "")
        .strip()
        for detok in dec
    ]


tokenizer_path = "dataset/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "dataset/tests/0000001.png"
img = Image.open(img_path).convert("RGB")
img_stem = Path(img_path).stem

s1 = time.perf_counter()
print("Loading model")
model_path = "outputs/Exp3/latest"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
print("Finished loading model.")

s2 = time.perf_counter()

pixel_values = processor(img, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)
print(pixel_values.shape)

s3 = time.perf_counter()

generated_ids = model.generate(pixel_values)
txt = token2str(generated_ids, tokenizer)
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("---------")
print(txt)
print(f"loading_model: {s2 - s1}s")
print(f"infer: {s3 - s2}s")
