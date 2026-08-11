"""
Character-level language model dataset for QIT-LM.

Sliding window: given ctx_len characters, predict the next one.
Vocab is built from the training text — no fixed tokenizer needed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

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

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.size = len(chars)
        self._c2i: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._i2c: dict[int, str] = {i: c for i, c in enumerate(chars)}

    @classmethod
    def from_chars(cls, chars: list[str]) -> "CharVocab":
        return cls(chars)

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        return cls.from_chars(sorted(set(text)))

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

    def __init__(self, token_ids, ctx_len: int, vocab: CharVocab):
        n_tokens = len(token_ids)
        if n_tokens <= ctx_len:
            raise ValueError(
                f"Text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} tokens, got {n_tokens}"
            )
        self.ids = token_ids
        self.ctx_len = ctx_len
        self.vocab = vocab

    @classmethod
    def from_string(cls, text: str, ctx_len: int) -> "CharLMDataset":
        vocab = CharVocab.from_text(text)
        ids = np.asarray(vocab.encode(text), dtype=np.uint32)
        return cls(ids, ctx_len, vocab)

    def __len__(self) -> int:
        return len(self.ids) - self.ctx_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = np.asarray(
            self.ids[idx : idx + self.ctx_len + 1],
            dtype=np.int64,
        )
        x = torch.from_numpy(sl[:-1].copy())
        y = torch.tensor(int(sl[-1]), dtype=torch.long)
        return x, y

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

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
    valid_text: str | None = None,
    corpus_path=None,
    valid_path=None,
    status_callback=None,
    progress_callback=None,
    max_train_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    from pathlib import Path

    from app.corpus_cache import get_or_encode_file, get_or_encode_text, get_or_encode_with_vocab

    def _make_train_loader(src_ds) -> DataLoader:
        # For very large datasets shuffle=True would call torch.randperm(N) which
        # allocates O(N*8) bytes (16 GB for a 2B-window corpus). Instead, pre-sample
        # max_train_samples indices via numpy, which is O(max_train_samples).
        if max_train_samples is not None and len(src_ds) > max_train_samples:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(src_ds), size=max_train_samples)
            src_ds = Subset(src_ds, idx.tolist())
        return DataLoader(src_ds, batch_size=batch_size, shuffle=True)

    def _status(msg: str, pct: float | None = None) -> None:
        if status_callback:
            status_callback(msg, pct)

    def _progress(msg: str, pct: float) -> None:
        if progress_callback:
            progress_callback(msg, pct)
        _status(msg, pct)

    if corpus_path is not None:
        path = Path(corpus_path)
        _status(f"Preparing corpus from {path.name}…", 0)
        vocab, train_ids, from_cache = get_or_encode_file(path, ctx_len, _progress)
        if from_cache:
            _status(f"Using cached encoding for {path.name} (memory-mapped)", 100)
        train_ds = CharLMDataset(train_ids, ctx_len, vocab)
    else:
        corpus = text if text is not None else DEMO_CORPUS
        if len(corpus) > 500_000:
            vocab, train_ids, from_cache = get_or_encode_text(
                corpus, ctx_len, _progress, label="pasted corpus"
            )
            train_ds = CharLMDataset(train_ids, ctx_len, vocab)
        else:
            _status(f"Scanning corpus ({len(corpus):,} characters)…", 0)
            _status("Building character vocabulary and encoding windows…", 5)
            train_ds = CharLMDataset.from_string(corpus, ctx_len)
            vocab = train_ds.vocab

    _status(
        f"Train set: {len(train_ds):,} windows ({vocab.size} unique chars)",
        100,
    )

    if valid_path is not None:
        vpath = Path(valid_path)
        _status(f"Loading validation file {vpath.name}…", 0)
        valid_ids, from_cache = get_or_encode_with_vocab(vpath, vocab, _progress)
        if from_cache:
            _status(f"Using cached validation encoding for {vpath.name}", 100)
        val_ds = CharLMDataset(valid_ids, ctx_len, vocab)
    elif valid_text is not None:
        _status(f"Encoding validation text ({len(valid_text):,} characters)…", 0)
        valid_ids = np.asarray(vocab.encode(valid_text), dtype=np.uint32)
        if len(valid_ids) <= ctx_len:
            raise ValueError(
                f"Validation text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} encoded chars, got {len(valid_ids)}"
            )
        val_ds = CharLMDataset(valid_ids, ctx_len, vocab)
    else:
        n_train = int(len(train_ds) * train_frac)
        n_val = len(train_ds) - n_train
        _status(
            f"Validation split: last {n_val:,} windows ({100 - train_frac * 100:.0f}% of train)",
            100,
        )
        train_loader = _make_train_loader(Subset(train_ds, range(n_train)))
        val_loader = DataLoader(
            Subset(train_ds, range(n_train, n_train + n_val)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        _status(f"Validation set: {n_val:,} windows", 100)
        return train_loader, val_loader, vocab

    _status(f"Validation set: {len(val_ds):,} windows", 100)
    train_loader = _make_train_loader(train_ds)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, vocab
