# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

img_path = "data/2016/UN_101_em_8.jpg"
img = Image.open(img_path).convert("RGB")
img.show()

img_stem = Path(img_path).stem

with open("data/2016/caption.txt", "r", encoding="utf-8") as label:
    for line in label.readlines():
        if img_stem == line.split("\t")[0]:
            print(line.split("\t")[1])


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
model = VisionEncoderDecoderModel.from_pretrained("outputs/checkpoint-1")

pixel_values = processor(img, return_tensors="pt").pixel_values
print(pixel_values.shape)


generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)


beam_output = model.generate(
    pixel_values,
    num_beams=10,
    early_stopping=True,
    num_return_sequences=5,
    max_length=490,
    # no_repeat_ngram_size = 3
)
print(processor.batch_decode(beam_output, skip_special_tokens=True)[0])
print(processor.batch_decode(beam_output, skip_special_tokens=True)[1])
print(processor.batch_decode(beam_output, skip_special_tokens=True)[2])
print(processor.batch_decode(beam_output, skip_special_tokens=True)[3])
print(processor.batch_decode(beam_output, skip_special_tokens=True)[4])


# max_length = 預測的字數
# no_repeat_ngram_size = 0(無窮大) 不出現重複的字幾次


sample_output = model.generate(pixel_values, do_sample=True, top_k=50)
print(processor.batch_decode(sample_output, skip_special_tokens=True)[0])
