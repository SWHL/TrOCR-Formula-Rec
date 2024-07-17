# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import pandas as pd
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

df = pd.read_table("./data/train/caption.txt", header=None)  # fwf
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)

df["file_name"] = df["file_name"].apply(lambda x: x + ".jpg")
df = df.dropna()

df2 = pd.read_table("./data/test/caption.txt", header=None)  # fwf
df2.rename(columns={0: "file_name", 1: "text"}, inplace=True)

df2["file_name"] = df2["file_name"].apply(lambda x: x + ".jpg")
df2 = df2.dropna()

train_df = df
test_df = df2
train_df = shuffle(train_df)

# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
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


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir="./data/train/", df=train_df, processor=processor)
eval_dataset = IAMDataset(root_dir="./data/test/", df=test_df, processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


encoding = train_dataset[0]
for k, v in encoding.items():
    print(k, v.shape)

image = Image.open(train_dataset.root_dir + train_df["file_name"][0]).convert("RGB")

labels = encoding["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)


model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 490
model.config.early_stopping = True

# model.config.no_repeat_ngram_size = 2
# model.config.length_penalty = 2.0
model.config.num_beams = 10

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=2,  # origin 8
    per_device_eval_batch_size=1,  # origin 8
    fp16=False,
    output_dir="outputs",
    logging_steps=2,
    save_steps=1,
    eval_steps=500,
    report_to=["tensorboard"],
    num_train_epochs=100,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()
