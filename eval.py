# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import random
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evaluate
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class IAMDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __getitem__(self, idx):
        file_name = self.data[idx]
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def __len__(self):
        return len(self.data)


def load_data(image_path, math_file):
    image_names = [f for f in sorted(os.listdir(image_path))]
    image_paths = [os.path.join(image_path, f) for f in image_names]

    math_gts = []
    with open(math_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            image_name = f"{i-1:07d}.png"
            if line.strip() and image_name in image_names:
                math_gts.append(line.strip())

    if len(image_paths) != len(math_gts):
        raise ValueError("The number of images does not match the number of formulas.")

    return image_paths, math_gts


def normalize_text(text):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = "[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", text)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    bleu = evaluate.load(
        "bleu", keep_in_memory=True, experiment_id=random.randint(1, 1e8)
    )
    bleu_results = bleu.compute(predictions=predictions, references=references)

    lev_dist = []
    for p, r in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return {"bleu": bleu_results["bleu"], "edit": sum(lev_dist) / len(lev_dist)}


def setup_seeds(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seeds()
    start = time.time()

    model_path = "outputs/checkpoint-27738"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    end1 = time.time()

    print(f"Load model: {end1 - start:.3f}s")

    val_names = [
        "Simple Print Expression(SPE)",
        "Complex Print Expression(CPE)",
        "Screen Capture Expression(SCE)",
        "Handwritten Expression(HWE)",
    ]
    image_paths = [
        "dataset/UniMER-Test/spe",
        "dataset/UniMER-Test/cpe",
        "dataset/UniMER-Test/sce",
        "dataset/UniMER-Test/hwe",
    ]
    math_files = [
        "dataset/UniMER-Test/spe.txt",
        "dataset/UniMER-Test/cpe.txt",
        "dataset/UniMER-Test/sce.txt",
        "dataset/UniMER-Test/hwe.txt",
    ]

    for val_name, image_path, math_file in zip(val_names, image_paths, math_files):
        image_list, math_gts = load_data(image_path, math_file)

        val_dataset = IAMDataset(image_list, processor)

        math_preds = []
        with torch.no_grad():
            for images in tqdm(val_dataset):
                images = images.to(device)
                generated_ids = model.generate(images)
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                math_preds.append(generated_text)

        # Compute BLEU/METEOR/EditDistance
        norm_gts = [normalize_text(gt) for gt in math_gts]
        norm_preds = [normalize_text(pred) for pred in math_preds]
        print(f"len_gts:{len(norm_gts)}, len_preds={len(norm_preds)}")
        print(f"norm_gts[0]:{norm_gts[0]}")
        print(f"norm_preds[0]:{norm_preds[0]}")

        p_scores = score_text(norm_preds, norm_gts)

        write_data = {
            "scores": p_scores,
            "text": [
                {"prediction": p, "reference": r} for p, r in zip(norm_preds, norm_gts)
            ],
        }

        score_table = []
        score_headers = ["bleu", "edit"]
        score_dirs = ["⬆", "⬇"]

        score_table.append([write_data["scores"][h] for h in score_headers])

        score_headers = [f"{h} {d}" for h, d in zip(score_headers, score_dirs)]

        end2 = time.time()

        print(f"Evaluation Set:{val_name}")
        print(f"Inference Time: {end2 - end1}s")
        print(tabulate(score_table, headers=[*score_headers]))
        print("=" * 100)


if __name__ == "__main__":
    main()
