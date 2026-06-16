"""
Structured task suite for QIT task-variety analysis.

Three tasks probing different structural properties:

  XORAtPositionsDataset:
      Predict x[i] XOR x[j] for fixed positions i,j.
      This is 2-position partial parity — a subset of global parity.
      Expected: QIT fast (quantum parity affinity), but faster than 4-bit?

  PartialParityDataset:
      Predict parity of first k tokens in an n-token sequence (k < n).
      For k=3, n=4: label = x[0]⊕x[1]⊕x[2], x[3] irrelevant.
      Expected: QIT fast; does convergence interpolate between k=2 and k=4?

  SequenceReversalDataset:
      Predict whether sequence is a palindrome (equals its own reverse).
      Positional (answer depends on position-pair comparisons, not XOR).
      Class-imbalanced: 4/16 = 25% palindromes for n_tokens=4.
      Expected: QIT should NOT be faster than classical here (negative control).
"""

import itertools

import torch
from torch.utils.data import DataLoader, Dataset


class XORAtPositionsDataset(Dataset):
    """
    Binary sequences of length n_tokens; label = x[pos_i] XOR x[pos_j].
    For n_tokens=4, pos_i=0, pos_j=1: 16 inputs, balanced (8 per class).
    """

    def __init__(self, n_tokens: int = 4, pos_i: int = 0, pos_j: int = 1):
        assert 0 <= pos_i < n_tokens and 0 <= pos_j < n_tokens and pos_i != pos_j
        seqs = list(itertools.product([0, 1], repeat=n_tokens))
        self.X = torch.tensor(seqs, dtype=torch.long)
        self.y = self.X[:, pos_i] ^ self.X[:, pos_j]
        self.pos_i = pos_i
        self.pos_j = pos_j

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class PartialParityDataset(Dataset):
    """
    Binary sequences of length n_tokens; label = parity of first k tokens.
    For n_tokens=4, k=3: label = x[0]⊕x[1]⊕x[2], x[3] ignored.
    16 inputs, balanced (8 per class).
    """

    def __init__(self, n_tokens: int = 4, k: int = 3):
        assert 1 <= k <= n_tokens
        seqs = list(itertools.product([0, 1], repeat=n_tokens))
        self.X = torch.tensor(seqs, dtype=torch.long)
        # XOR reduction over first k positions
        self.y = self.X[:, 0].clone()
        for col in range(1, k):
            self.y = self.y ^ self.X[:, col]
        self.k = k

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class SequenceReversalDataset(Dataset):
    """
    Binary sequences of length n_tokens; label = 1 if palindrome, 0 otherwise.
    For n_tokens=4: 4/16 are palindromes (0000, 0110, 1001, 1111).
    Class-imbalanced: random baseline = 75% accuracy.

    Positional task — palindrome check requires comparing x[i] vs x[n-1-i],
    not any global parity or sum statistic.
    """

    def __init__(self, n_tokens: int = 4):
        seqs = list(itertools.product([0, 1], repeat=n_tokens))
        self.X = torch.tensor(seqs, dtype=torch.long)
        self.y = torch.tensor(
            [int(list(s) == list(reversed(s))) for s in seqs], dtype=torch.long
        )
        self.n_palindromes = int(self.y.sum().item())

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── DataLoader factories ───────────────────────────────────────────────────────

def make_xor_positions_loaders(
    n_tokens: int = 4,
    pos_i: int = 0,
    pos_j: int = 1,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    ds = XORAtPositionsDataset(n_tokens=n_tokens, pos_i=pos_i, pos_j=pos_j)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_partial_parity_loaders(
    n_tokens: int = 4,
    k: int = 3,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    ds = PartialParityDataset(n_tokens=n_tokens, k=k)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_sequence_reversal_loaders(
    n_tokens: int = 4,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    ds = SequenceReversalDataset(n_tokens=n_tokens)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
