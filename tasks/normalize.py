"""
Corpus text normalization for QIT-LM.

With vocab=228 the quantum decoder (12 inputs → 228 classes) can't converge.
Normalizing to ~35 chars (a-z + space + basic punctuation) makes the task
tractable for a 3-4k parameter model.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize_corpus(text: str, min_char_freq: int = 10) -> str:
    """
    Reduce character vocabulary for better QIT-LM training.

    Steps:
      1. NFKD decomposition + strip combining marks  (café → cafe, naïve → naive)
      2. Lowercase
      3. Non-ASCII → space
      4. Collapse whitespace runs (preserve newlines)
      5. Drop characters appearing fewer than min_char_freq times

    Typical result: vocab 200+ → 35-45 chars.
    Set min_char_freq=0 to skip the frequency filter (useful for val set).
    """
    # Decompose and strip accents / diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    text = text.lower()

    # Non-ASCII (after decomposition) → space
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)

    # Collapse horizontal whitespace; preserve newlines as single \n
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Frequency filter — drop rare chars so the model doesn't waste capacity on them
    if min_char_freq > 1:
        freq = Counter(text)
        keep = {c for c, n in freq.items() if n >= min_char_freq} | {" ", "\n"}
        text = "".join(c if c in keep else " " for c in text)
        text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
