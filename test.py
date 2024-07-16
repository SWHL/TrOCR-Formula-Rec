# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import evaluate
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

df = pd.read_table("./data/2014/caption.txt", header=None)  # 2014
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df["file_name"] = df["file_name"].apply(lambda x: x + ".jpg")
df = df.dropna()
df.head()


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding

    def __len__(self):
        return len(self.df)


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
test_dataset = IAMDataset(root_dir="./data/2014/", df=df, processor=processor)

test_dataloader = DataLoader(test_dataset, batch_size=1)
batch = next(iter(test_dataloader))
for k, v in batch.items():
    print(k, v.shape)

labels = batch["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.batch_decode(labels, skip_special_tokens=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("outputs/checkpoint-1")
model.to(device)

cer_metric = evaluate.load("cer")

print("Running evaluation...")

total = 0
pred_label = 0

for batch in tqdm(test_dataloader):
    pixel_values = batch["pixel_values"].to(device)
    outputs = model.generate(pixel_values)

    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    if pred_str == label_str:
        pred_label += 1
    total += 1

    # add batch to metric
    cer_metric.add_batch(predictions=pred_str, references=label_str)

Accuracy_score = pred_label / total
final_score = cer_metric.compute()

print("Character error rate on test set:", final_score)
print("Accuracy rate on test set:", Accuracy_score)
