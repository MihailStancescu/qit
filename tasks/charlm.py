"""
Character-level language model dataset for QIT-LM.

Sliding window: given ctx_len characters, predict the next one.
Vocab is built from the training text — no fixed tokenizer needed.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Short, repetitive demo corpus — good for tiny QIT-LM to memorize patterns.
DEMO_CORPUS = (
    "the quick brown fox jumps over the lazy dog "
    "the fox ran fast and the dog ran slow "
    "a quick brown dog jumps over the lazy fox "
    "the cat sat on the mat the rat sat on the cat "
    "one fish two fish red fish blue fish "
    "the cat the dog the fox the rat the hog "
    "to be or not to be that is the question "
    "all that glitters is not gold "
)


class CharVocab:
    """Bijective char ↔ integer mapping built from a text corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.chars = chars
        self.size = len(chars)
        self._c2i: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._i2c: dict[int, str] = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2c.get(i, "?") for i in ids)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        preview = "".join(self.chars[:20]) + ("…" if self.size > 20 else "")
        return f"CharVocab(size={self.size}, chars={preview!r})"


class CharLMDataset(Dataset):
    """
    Sliding-window character-level LM dataset.

    Each sample: (x, y) where
        x = token_ids[i : i + ctx_len]   shape (ctx_len,)
        y = token_ids[i + ctx_len]        scalar  (next char id)
    """

    def __init__(self, token_ids: list[int], ctx_len: int, vocab: CharVocab):
        if len(token_ids) <= ctx_len:
            raise ValueError(
                f"Text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} chars, got {len(token_ids)}"
            )
        self.ids = token_ids
        self.ctx_len = ctx_len
        self.vocab = vocab

    @classmethod
    def from_string(cls, text: str, ctx_len: int) -> "CharLMDataset":
        vocab = CharVocab(text)
        return cls(vocab.encode(text), ctx_len, vocab)

    @classmethod
    def from_file(cls, path: str, ctx_len: int) -> "CharLMDataset":
        return cls.from_string(open(path).read(), ctx_len)

    @classmethod
    def demo(cls, ctx_len: int = 8) -> "CharLMDataset":
        return cls.from_string(DEMO_CORPUS, ctx_len)

    def __len__(self) -> int:
        return len(self.ids) - self.ctx_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.ids[idx : idx + self.ctx_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx + self.ctx_len], dtype=torch.long)
        return x, y

    def __repr__(self) -> str:
        return (
            f"CharLMDataset(ctx_len={self.ctx_len}, "
            f"samples={len(self)}, {self.vocab})"
        )


def make_charlm_loaders(
    text: str | None = None,
    ctx_len: int = 8,
    batch_size: int = 8,
    train_frac: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    """
    Build train/val DataLoaders from text.
    Split is positional (first 90% train, last 10% val) — not random —
    so val sees text the model hasn't trained on.

    Returns:
        (train_loader, val_loader, vocab)
    """
    corpus = text if text is not None else DEMO_CORPUS
    ds = CharLMDataset.from_string(corpus, ctx_len)
    vocab = ds.vocab

    n_train = int(len(ds) * train_frac)
    n_val = len(ds) - n_train

    # Positional split: first n_train windows for train, rest for val.
    from torch.utils.data import Subset
    train_ds = Subset(ds, range(n_train))
    val_ds = Subset(ds, range(n_train, n_train + n_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab
