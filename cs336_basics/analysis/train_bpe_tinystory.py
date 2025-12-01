"""
# train small data set
uv run cs336_basics/analysis/train_bpe_tinystory.py \
  --input_path=tests/fixtures/small.txt \
  --special_tokens="<|endoftext|>" \
  --vocab_size=263

# train 5MiB data set
uv run cs336_basics/analysis/train_bpe_tinystory.py \
  --input_path=tests/fixtures/tinystories_sample_5M.txt \
  --special_tokens="<|endoftext|>"

# train 2GiB data set
uv run cs336_basics/analysis/train_bpe_tinystory.py \
  --input_path=cs336_basics/analysis/data/TinyStoriesV2-GPT4-train.txt \
  --special_tokens="<|endoftext|>"\
  > /tmp/TinyStoriesV2-BPE-train.out
"""
from cs336_basics.bpe.trainer import BpeTrainer

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", help="path to input BPE training corpus.")
    parser.add_argument(
        "--vocab_size", type=int, default=500, help="vocab size.")
    parser.add_argument(
        "--special_tokens", default='', help="special tokens.")

    args = parser.parse_args()
    print(f'training file: {args.input_path}')

    bpe_trainer = BpeTrainer(
        args.vocab_size, 
        args.special_tokens.split(','))
    
    bpe_trainer.train(args.input_path)
    bpe_trainer.persist(
        vocab_path='/tmp/vocab.bin',
        merges_path='/tmp/merges.bin',
    )

    print(bpe_trainer)
