# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from transformers import PreTrainedTokenizerFast

tokenizer_path = "dataset/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

eqs = "( \mathrm { P } < 0 . 0 0 0 1 ,"
tok = tokenizer(list(eqs), return_token_type_ids=False)

print("ok")
