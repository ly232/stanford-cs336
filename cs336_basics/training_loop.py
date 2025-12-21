"""Script to run training loop."""

from cs336_basics.model.transformer_language_model import TransformerLanguageModel
from cs336_basics.data.data_loader import DataLoader
from cs336_basics.training.adamw_optimizer import AdamW
from cs336_basics.training.utils import cross_entropy
from cs336_basics.data.checkpoint import Checkpoint

import numpy as np
import torch


def main():
    # Model hyperparameters.
    vocab_size = 1024
    context_length = 8
    d_model = 16
    num_layers = 2
    num_heads = 4
    d_ff = 64
    rope_theta = 10000.0

    # Optimizer hyperparameters.
    lr = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8

    # Data loading params.
    batch_size = 32
    device = "cpu"
    num_iters = 1000
    tokens_path = "data/tokens.txt"
    checkpoint_path = "checkpoints/model.chkpt"

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
    dataset = np.memmap(tokens_path, dtype=np.int16, mode="r")
    for step in range(num_iters):
        x, y = data_loader.get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

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
