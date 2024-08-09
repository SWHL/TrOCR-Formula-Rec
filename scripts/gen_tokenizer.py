from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def generate_tokenizer(equations, output, vocab_size):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[BOS]", "[EOS]"],
        vocab_size=vocab_size,
        show_progress=True,
    )
    tokenizer.train(equations, trainer)
    tokenizer.save(path=output, pretty=False)


equations = ["dataset/UniMER-1M/train.txt"]
output = "tokenizer.json"
vocab_size = 8000

generate_tokenizer(equations, output, vocab_size)
