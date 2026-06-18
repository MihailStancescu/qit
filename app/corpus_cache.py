"""
Disk cache for encoded character-LM corpora.

Keyed by source identity (uploaded file name + size + mtime, or content hash for
pasted text) and ctx_len so repeat training runs skip the expensive encode step.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from tasks.charlm import CharVocab

CACHE_DIR = Path(__file__).parent / "data" / "corpus_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CHUNK = 8_000_000  # chars per read / progress tick


def _cache_dir(key: str) -> Path:
    d = CACHE_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key_for_file(path: Path, ctx_len: int) -> str:
    stat = path.stat()
    raw = f"file|{path.name}|{stat.st_size}|{int(stat.st_mtime_ns)}|{ctx_len}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def cache_key_for_text(text: str, ctx_len: int, label: str = "paste") -> str:
    head = text[:8192]
    raw = f"text|{label}|{len(text)}|{head}|{ctx_len}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:20]


def _meta_path(key: str) -> Path:
    return _cache_dir(key) / "meta.json"


def _vocab_path(key: str) -> Path:
    return _cache_dir(key) / "vocab.pkl"


def _tokens_path(key: str) -> Path:
    return _cache_dir(key) / "token_ids.npy"


def load_encoded_corpus(key: str) -> tuple[CharVocab, np.ndarray, dict] | None:
    """Return (vocab, token_ids, meta) if a valid cache entry exists."""
    meta_p, vocab_p, tok_p = _meta_path(key), _vocab_path(key), _tokens_path(key)
    if not (meta_p.exists() and vocab_p.exists() and tok_p.exists()):
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        with vocab_p.open("rb") as f:
            vocab: CharVocab = pickle.load(f)
        token_ids = np.load(tok_p, mmap_mode="r")
        return vocab, token_ids, meta
    except (OSError, json.JSONDecodeError, pickle.UnpicklingError, ValueError):
        return None


def save_encoded_corpus(
    key: str,
    vocab: CharVocab,
    token_ids: np.ndarray,
    *,
    source_name: str,
    n_chars: int,
    ctx_len: int,
) -> None:
    _cache_dir(key)
    tok_path = _tokens_path(key)
    n = len(token_ids)
    mm = np.lib.format.open_memmap(
        tok_path, mode="w+", dtype=np.uint32, shape=(max(n, 1),)
    )
    chunk = 4_000_000
    for i in range(0, n, chunk):
        mm[i : i + chunk] = token_ids[i : i + chunk]
    mm.flush()
    with _vocab_path(key).open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "source_name": source_name,
        "n_chars": n_chars,
        "n_tokens": n,
        "ctx_len": ctx_len,
        "vocab_size": vocab.size,
        "created_at": time.time(),
    }
    _meta_path(key).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _progress(callback, message: str, pct: float) -> None:
    if callback:
        callback(message, pct)


def vocab_from_path(path: Path, progress_callback=None) -> CharVocab:
    """Stream a file to build a character vocabulary without loading it all at once."""
    chars: set[str] = set()
    size = path.stat().st_size
    read = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            chars.update(chunk)
            read += len(chunk)
            if size > 0:
                _progress(
                    progress_callback,
                    f"Scanning vocabulary in {path.name}",
                    min(99.0, 100.0 * read / max(size, 1)),
                )
    return CharVocab.from_chars(sorted(chars))


def vocab_from_text(text: str) -> CharVocab:
    return CharVocab.from_chars(sorted(set(text)))


def count_tokens(path: Path, vocab: CharVocab) -> int:
    c2i = vocab._c2i
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            for c in chunk:
                if c in c2i:
                    n += 1
    return n


def encode_path(
    path: Path,
    vocab: CharVocab,
    progress_callback=None,
    label: str | None = None,
    out_path: Path | None = None,
) -> np.ndarray:
    """Stream-encode a text file; optionally write directly to a memmap file."""
    c2i = vocab._c2i
    label = label or path.name
    file_size = path.stat().st_size

    if out_path is not None:
        n_tokens = count_tokens(path, vocab)
        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.uint32, shape=(max(n_tokens, 1),)
        )
        pos = 0
        processed = 0
        with path.open(encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                for c in chunk:
                    if c in c2i:
                        mm[pos] = c2i[c]
                        pos += 1
                processed += len(chunk)
                _progress(
                    progress_callback,
                    f"Encoding {label}",
                    min(99.0, 100.0 * processed / max(file_size, 1)),
                )
        mm.flush()
        return np.load(out_path, mmap_mode="r")

    # In-memory fallback for small files.
    est = file_size
    out = np.empty(est, dtype=np.uint32)
    pos = 0
    processed = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            for c in chunk:
                if c in c2i:
                    if pos >= len(out):
                        out = np.resize(out, len(out) + _CHUNK)
                    out[pos] = c2i[c]
                    pos += 1
            processed += len(chunk)
            _progress(
                progress_callback,
                f"Encoding {label}",
                min(99.0, 100.0 * processed / max(file_size, 1)),
            )
    return out[:pos]


def encode_text(
    text: str,
    vocab: CharVocab,
    progress_callback=None,
    label: str = "corpus",
) -> np.ndarray:
    """Encode in-memory text with progress ticks."""
    c2i = vocab._c2i
    n = len(text)
    out = np.empty(n, dtype=np.uint32)
    pos = 0
    for start in range(0, n, _CHUNK):
        chunk = text[start : start + _CHUNK]
        for c in chunk:
            if c in c2i:
                out[pos] = c2i[c]
                pos += 1
        _progress(
            progress_callback,
            f"Encoding {label}",
            min(99.0, 100.0 * (start + len(chunk)) / max(n, 1)),
        )
    return out[:pos]


def cache_key_for_file_with_vocab(path: Path, vocab: CharVocab) -> str:
    stat = path.stat()
    raw = f"valid|{path.name}|{stat.st_size}|{int(stat.st_mtime_ns)}|v{vocab.size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def get_or_encode_with_vocab(
    path: Path,
    vocab: CharVocab,
    progress_callback=None,
) -> tuple[np.ndarray, bool]:
    """Encode path with an existing vocab; cache to disk as mmap."""
    key = cache_key_for_file_with_vocab(path, vocab)
    tok_p = _tokens_path(key)
    meta_p = _meta_path(key)
    if tok_p.exists() and meta_p.exists():
        if progress_callback:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            progress_callback(
                f"Loaded cached validation encoding for {path.name} "
                f"({meta.get('n_tokens', 0):,} tokens)",
                100.0,
            )
        return np.load(tok_p, mmap_mode="r"), True

    _cache_dir(key)
    token_ids = encode_path(path, vocab, progress_callback, path.name, out_path=tok_p)
    meta = {
        "source_name": path.name,
        "n_tokens": int(len(token_ids)),
        "vocab_size": vocab.size,
        "created_at": time.time(),
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return token_ids, False


def get_or_encode_file(
    path: Path,
    ctx_len: int,
    progress_callback=None,
) -> tuple[CharVocab, np.ndarray, bool]:
    """
    Load cached token ids for path or build + cache them.
    Returns (vocab, token_ids, from_cache).
    """
    key = cache_key_for_file(path, ctx_len)
    cached = load_encoded_corpus(key)
    if cached is not None:
        vocab, token_ids, meta = cached
        if progress_callback:
            progress_callback(
                f"Loaded cached encoding for {meta.get('source_name', path.name)} "
                f"({meta.get('n_tokens', len(token_ids)):,} tokens)",
                100.0,
            )
        return vocab, token_ids, True

    if progress_callback:
        progress_callback(f"Building vocabulary from {path.name}…", 0.0)
    vocab = vocab_from_path(path, progress_callback)
    tok_path = _tokens_path(key)
    token_ids = encode_path(path, vocab, progress_callback, path.name, out_path=tok_path)
    with _vocab_path(key).open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "source_name": path.name,
        "n_chars": int(path.stat().st_size),
        "n_tokens": int(len(token_ids)),
        "ctx_len": ctx_len,
        "vocab_size": vocab.size,
        "created_at": time.time(),
    }
    _meta_path(key).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(f"Cached encoding for {path.name}", 100.0)
    return vocab, token_ids, False


def get_or_encode_text(
    text: str,
    ctx_len: int,
    progress_callback=None,
    label: str = "pasted corpus",
) -> tuple[CharVocab, np.ndarray, bool]:
    key = cache_key_for_text(text, ctx_len, label)
    cached = load_encoded_corpus(key)
    if cached is not None:
        vocab, token_ids, meta = cached
        if progress_callback:
            progress_callback(
                f"Loaded cached encoding ({meta.get('n_tokens', len(token_ids)):,} tokens)",
                100.0,
            )
        return vocab, token_ids, True

    if progress_callback:
        progress_callback(f"Building vocabulary ({len(text):,} characters)…", 0.0)
    vocab = vocab_from_text(text)
    token_ids = encode_text(text, vocab, progress_callback, label)
    save_encoded_corpus(
        key,
        vocab,
        token_ids,
        source_name=label,
        n_chars=len(text),
        ctx_len=ctx_len,
    )
    return vocab, token_ids, False
