"""
Parity task: predict XOR (sum mod 2) of a binary sequence.

Why parity is a good first benchmark for QIT:
- Requires attending to every position simultaneously — no shortcut via local patterns.
- Classical attention can solve it, but requires O(n) operations.
- Quantum interference is theoretically well-suited: the parity function maps
  naturally to the phase kickback structure of Grover/Deutsch-Jozsa circuits.
- For n_tokens=4 there are only 16 inputs (perfectly balanced: 8 even, 8 odd),
  making convergence easy to verify and visualize.
"""

import itertools

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ParityDataset(Dataset):
    """
    Binary sequence → parity label (0=even, 1=odd count of 1s).

    Two modes:
    - Enumerated (size=None): all 2^n_tokens inputs. Use for n_tokens ≤ 5.
    - Sampled   (size=int):   random draws with replacement. Use for training
      curves or to simulate a streaming data regime.
    """

    def __init__(
        self,
        n_tokens: int = 4,
        size: int | None = None,
        seed: int = 42,
    ):
        self.n_tokens = n_tokens

        if size is None:
            seqs = list(itertools.product([0, 1], repeat=n_tokens))
            self.X = torch.tensor(seqs, dtype=torch.long)
        else:
            rng = torch.Generator().manual_seed(seed)
            self.X = torch.randint(0, 2, (size, n_tokens), generator=rng)

        self.y = self.X.sum(dim=1) % 2  # XOR parity

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def class_balance(self) -> dict[str, int]:
        counts = self.y.bincount(minlength=2)
        return {"even_parity": counts[0].item(), "odd_parity": counts[1].item()}

    def __repr__(self) -> str:
        bal = self.class_balance()
        return (
            f"ParityDataset(n_tokens={self.n_tokens}, "
            f"size={len(self)}, balance={bal})"
        )


# ── loader factories ──────────────────────────────────────────────────────────

def make_parity_loaders(
    n_tokens: int = 4,
    train_size: int = 800,
    test_size: int = 200,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Sampled train/test split. Good for convergence curves and overfitting analysis.
    train and test sets are drawn independently so there is deliberate overlap —
    the task has only 2^n unique inputs, so this is intentional.
    """
    train_ds = ParityDataset(n_tokens=n_tokens, size=train_size, seed=seed)
    test_ds  = ParityDataset(n_tokens=n_tokens, size=test_size,  seed=seed + 99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def make_full_parity_loaders(
    n_tokens: int = 4,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Enumerates all 2^n_tokens inputs and splits 80/20.
    Use this to verify that QIT-0 can perfectly memorise the parity function.
    For n_tokens=4: 16 total → 13 train, 3 test.
    """
    full_ds = ParityDataset(n_tokens=n_tokens, size=None)
    n_total = len(full_ds)
    n_test  = max(1, n_total // 5)
    n_train = n_total - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def make_memorization_loaders(
    n_tokens: int = 4,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """
    Uses all 2^n_tokens inputs for BOTH train and test.

    This is a memorization test: 100% accuracy means the model has learned
    the complete parity function over the entire input space, not just a
    subset. The correct experiment for comparing sample efficiency across
    architectures when 2^n is small enough to enumerate.
    """
    ds = ParityDataset(n_tokens=n_tokens, size=None)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
