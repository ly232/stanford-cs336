"""Script to run training loop.

uv run python cs336_basics/training_loop.py
"""

from pathlib import Path
from tests.common import FIXTURES_PATH

from cs336_basics.bpe.trainer import BpeTrainer
from cs336_basics.bpe.codec import BpeCodec
from cs336_basics.model.transformer_language_model import TransformerLanguageModel
from cs336_basics.data.data_loader import DataLoader
from cs336_basics.training.adamw_optimizer import AdamW
from cs336_basics.training.utils import cross_entropy
from cs336_basics.data.checkpoint import Checkpoint

import numpy as np
import torch
import pickle
import tqdm


def main():
    # Model hyperparameters.
    vocab_size = 10000
    context_length = 254
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0
    max_num_tokens = 4e7

    # Optimizer hyperparameters.
    lr = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8

    # Train BPE.
    tokens_path = Path("./cs336_basics/data/tokens.npy")
    bpe_vocab_path = Path("./cs336_basics/data/bpe_vocab.pkl")
    bpe_merges_path = Path("./cs336_basics/data/bpe_merges.pkl")
    tinystories_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    if not bpe_vocab_path.exists() or not bpe_merges_path.exists():
        bpe_trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
        vocab, merges = bpe_trainer.train(input_path=tinystories_path)
        bpe_trainer.persist(
            vocab_path="./cs336_basics/data/bpe_vocab.pkl",
            merges_path="./cs336_basics/data/bpe_merges.pkl",
        )
    else:
        vocab = pickle.load(open(bpe_vocab_path, "rb"))
        merges = pickle.load(open(bpe_merges_path, "rb"))
    bpe_codec = BpeCodec(vocab, merges)
    if not tokens_path.exists():
        with open(tinystories_path, "r") as f:
            text = f.read()
            tokens = []
            for tok in tqdm.tqdm(
                bpe_codec.encode_iterable(text), desc="Encoding", unit="tok"
            ):
                tokens.append(tok)
            # ATTN! dtype is crucial here. token ids must be non-negative to be indexble,
            # so they must be uint. Picking uint16 here because vocab size is 10,000 < 2^16.
            np.save(tokens_path, np.array(tokens, dtype=np.uint16))

    # Data loading params.
    batch_size = 32
    device = "cpu"
    num_iters = int(max_num_tokens // (batch_size * context_length))
    checkpoint_path = "./cs336_basics/data/model.chkpt"

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

    data_loader = DataLoader()
    checkpoint = Checkpoint()
    dataset = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    for step in tqdm.tqdm(range(num_iters), desc="Training", unit="step"):
        x, y = data_loader.get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        ##
        # debugging commands: peek into loaded x's and decode to make sure content makes sense
        # if x.max() >= vocab_size:
        #     print(f"Step: {step}, x.shape: {x.shape}, xmin: {x.min()}, xmax: {x.max()}")
        #     print(x)
        #     for token_ids in x:
        #         if token_ids.max() >= vocab_size:
        #             print("----- Decoding unexpected token ids -----")
        #             print(token_ids)
        #             decoded_text = bpe_codec.decode(token_ids.tolist())
        #             print(decoded_text)
        #             print("==========")
        # print("#########")

        # Forward pass
        logits = model(x)
        assert logits.shape == (batch_size, context_length, vocab_size)
        loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        # Backward pass + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item()}")

    # Save final checkpoint
    checkpoint.save_checkpoint(model, optimizer, num_iters - 1, checkpoint_path)


if __name__ == "__main__":
    main()
