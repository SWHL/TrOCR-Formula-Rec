# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from transformers import PreTrainedTokenizerFast

tokenizer_path = "dataset/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

eqs = "( \mathrm { P } < 0 . 0 0 0 1 ,"
tokenizer.add_special_tokens(
    {"pad_token": "[PAD]", "eos_token": "[EOS]", "bos_token": "[BOS]"}
)
tok = tokenizer([eqs], padding="max_length", max_length=512, truncation=True)
print("ok")
