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
        if len(token_ids) <= ctx_len:
            raise ValueError(
                f"Text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} tokens, got {len(token_ids)}"
            )
        self.ids = token_ids
        self.ctx_len = ctx_len
        self.vocab = vocab

    @classmethod
    def from_string(cls, text: str, ctx_len: int) -> "CharLMDataset":
        vocab = CharVocab.from_text(text)
        return cls(vocab.encode(text), ctx_len, vocab)

    @classmethod
    def from_file(cls, path: str, ctx_len: int) -> "CharLMDataset":
        with open(path, encoding="utf-8", errors="replace") as f:
            return cls.from_string(f.read(), ctx_len)

    @classmethod
    def demo(cls, ctx_len: int = 8) -> "CharLMDataset":
        return cls.from_string(DEMO_CORPUS, ctx_len)

    def __len__(self) -> int:
        return len(self.ids) - self.ctx_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = self.ids[idx : idx + self.ctx_len + 1]
        x = torch.as_tensor(sl[:-1], dtype=torch.long)
        y = torch.as_tensor(sl[-1], dtype=torch.long)
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
    valid_text: str | None = None,
    corpus_path=None,
    valid_path=None,
    status_callback=None,
    progress_callback=None,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    """
    Build train/val DataLoaders.

    If valid_text is provided it is used as the validation set and the
    train corpus is used in full (train_frac is ignored). The vocab is
    built from the train text; characters in valid_text not seen during
    training are silently filtered by CharVocab.encode().

    Otherwise a positional split (first train_frac of windows) is used.

    Returns:
        (train_loader, val_loader, vocab)
    """
    from pathlib import Path

    from app.corpus_cache import (
        encode_path,
        encode_text,
        get_or_encode_file,
        get_or_encode_text,
    )
    from torch.utils.data import Subset

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
            _status(f"Using cached encoding for {path.name}", 100)
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
        f"Train set ready: {len(train_ds):,} windows, {vocab.size} unique characters",
        100,
    )

    if valid_path is not None:
        vpath = Path(valid_path)
        _status(f"Encoding validation file {vpath.name}…", 0)
        valid_ids = encode_path(vpath, vocab, _progress, vpath.name)
        if len(valid_ids) <= ctx_len:
            raise ValueError(
                f"Validation text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} encoded chars, got {len(valid_ids)}"
            )
        val_ds = CharLMDataset(valid_ids, ctx_len, vocab)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        _status(f"Validation set ready: {len(val_ds):,} windows", 100)
    elif valid_text is not None:
        _status(f"Encoding validation corpus ({len(valid_text):,} characters)…", 0)
        valid_ids = encode_text(valid_text, vocab, _progress, "validation")
        if len(valid_ids) <= ctx_len:
            raise ValueError(
                f"Validation text too short for ctx_len={ctx_len}: "
                f"need >{ctx_len} encoded chars, got {len(valid_ids)}"
            )
        val_ds = CharLMDataset(valid_ids, ctx_len, vocab)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        _status(f"Validation set ready: {len(val_ds):,} windows", 100)
    else:
        n_train = int(len(train_ds) * train_frac)
        n_val   = len(train_ds) - n_train
        _status(f"Splitting data: {n_train:,} train / {n_val:,} val windows", 100)
        train_loader = DataLoader(Subset(train_ds, range(n_train)),           batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(train_ds, range(n_train, n_train + n_val)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab
