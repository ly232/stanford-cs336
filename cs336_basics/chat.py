"""A chat interface for interacting with the trained language model.

uv run python cs336_basics/chat.py
"""

from cs336_basics import decoding
from cs336_basics.model.transformer_language_model import TransformerLanguageModel
from cs336_basics.bpe.codec import BpeCodec
from cs336_basics.training.adamw_optimizer import AdamW
from cs336_basics.data.checkpoint import Checkpoint
from hyperparameters import *
from pathlib import Path

import pickle
import torch
import tqdm


def chat_loop(model: TransformerLanguageModel, bpe_codec: BpeCodec):
    # Example:
    # Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much.
    prompt = input("Please enter your starting message: ")
    tokens = bpe_codec.encode(prompt)
    for token in tqdm.tqdm(range(1000), desc="Generating text", unit="token"):
        next_token = (
            decoding.next_token(
                model, torch.tensor(tokens[-context_length:]).unsqueeze(0)
            )
            .squeeze(0)
            .tolist()
        )
        if bpe_codec.decode(next_token) == "<|endoftext|>":
            print("(reached EOS. stopping.)")
            break
        tokens.extend(next_token)
    response = bpe_codec.decode(tokens)
    print(f"[Model response]:\n{response}")


if __name__ == "__main__":
    # Load model.
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )
    checkpoint = Checkpoint()
    checkpoint.load_checkpoint(
        src="./cs336_basics/data/model.chkpt", model=model, optimizer=optimizer
    )

    # Load BPE codec.
    bpe_vocab_path = Path("./cs336_basics/data/bpe_vocab.pkl")
    bpe_merges_path = Path("./cs336_basics/data/bpe_merges.pkl")
    vocab = pickle.load(open(bpe_vocab_path, "rb"))
    merges = pickle.load(open(bpe_merges_path, "rb"))
    bpe_codec = BpeCodec(vocab, merges)

    # kick start the chat loop.
    chat_loop(model, bpe_codec)
