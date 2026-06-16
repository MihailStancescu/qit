"""
First-token detection: predict x[0] (value of the first token).

Serves as a negative-control benchmark for QIT — a task with:
  - Positional structure (answer depends on position 0 only)
  - No global quantum phase structure (no XOR of all tokens)
  - Tests whether QIT's parity advantage is task-specific or universal

If QIT is fast here too → 16-input memorisation is trivially fast for
any expressive model, and the parity result needs recontextualisation.
If QIT is not faster than classical here → the parity advantage reflects
genuine quantum affinity, not circuit expressibility on small datasets.
"""

import itertools

import torch
from torch.utils.data import DataLoader, Dataset


class FirstTokenDataset(Dataset):
    """
    Binary sequences of length n_tokens; label = value of first token.
    For n_tokens=4: 16 total inputs, perfectly balanced (8 each class).
    """

    def __init__(self, n_tokens: int = 4):
        seqs = list(itertools.product([0, 1], repeat=n_tokens))
        self.X = torch.tensor(seqs, dtype=torch.long)
        self.y = self.X[:, 0]  # label = first token

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_first_token_loaders(
    n_tokens: int = 4,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """
    Memorisation protocol: all 2^n_tokens inputs for both train and test.
    """
    ds = FirstTokenDataset(n_tokens=n_tokens)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
