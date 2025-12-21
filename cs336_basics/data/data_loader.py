"""Data loader to load corpus data to feed into model training."""

import numpy.typing as npt
import random
import torch


class DataLoader:

    def __init__(self):
        pass

    def get_batch(
        self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ):
        """
        Returns:
            Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
            is the sampled input sequences, and the second tuple item is the corresponding
            language modeling labels.
        """
        xs, ys = [], []
        for i in range(batch_size):
            start_idx = random.randint(0, len(dataset) - context_length - 1)
            x = torch.from_numpy(dataset[start_idx : start_idx + context_length]).long()
            y = torch.from_numpy(
                dataset[start_idx + 1 : start_idx + context_length + 1]
            ).long()
            xs.append(x.unsqueeze(0))
            ys.append(y.unsqueeze(0))
        return torch.cat(xs, dim=0).long().to(device), torch.cat(ys, dim=0).long().to(
            device
        )
