"""Utilities to decode from a trained language model."""

import torch

from cs336_basics.model.utils import softmax


def next_token(model, prompt_tokens, max_length, temperature=1.0, nucleus_p=1.0):
    batch_size, seq_length = prompt_tokens.shape
    logits = model(prompt_tokens) / temperature
    _, _, vocab_size = logits.shape  # batch_size, seq_length, vocab_size
    probs = softmax(logits[:, -1, :], dim=-1)  # batch_size, vocab_size
    sorted_probs, sorted_indices = torch.sort(
        probs, descending=True, dim=-1
    )  # batch_size, vocab_size

    # vectorized way to compute nucleus mask
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    crossed = cumulative_probs >= nucleus_p
    # find first index where cumulative_probs >= nucleus_p, keep everything in sorted
    # probs up to and including this index.
    cutoff_index = crossed.float().argmax(dim=-1)  # batch_size
    nucleus_mask = torch.arange(vocab_size).to(device=prompt_tokens.device)
    # unsqueeze to broadcast:
    # nucleus_mask: before: (vocab_size,) after: (1, vocab_size)
    # cutoff_index: before: (batch_size,) after: (batch_size, 1)
    # nucleus_mask: after broadcast: (batch_size, vocab_size)
    nucleus_mask = nucleus_mask.unsqueeze(0) <= cutoff_index.unsqueeze(
        1
    )  # batch_size, vocab_size

    probs = torch.zeros_like(probs)
    probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * nucleus_mask)
    # Renormalize probs against last dim.
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # sample from the distribution
    next_tokens = torch.multinomial(probs, num_samples=1)  # batch_size
    return next_tokens
