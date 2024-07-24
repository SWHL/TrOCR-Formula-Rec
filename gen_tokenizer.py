from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

vocab_size = 8000
equation_path = ["dataset/UniMER-1M/train.txt"]
save_path = "tokenizer.json"

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(
    special_tokens=["[PAD]", "[BOS]", "[EOS]"],
    vocab_size=vocab_size,
    show_progress=True,
)
tokenizer.train(equation_path, trainer)
tokenizer.save(path=save_path, pretty=False)
