"""Shared helpers for offline evaluation harnesses."""

from __future__ import annotations

import string
import unicodedata

PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation + "„“‚‘’«»…–—"})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(PUNCT_TRANSLATION)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return WER as errors / reference-token count.

    >>> round(word_error_rate("eins zwei drei", "eins drei"), 3)
    0.333
    >>> word_error_rate("eins zwei", "eins zwei")
    0.0
    """
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_token in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (ref_token != hyp_token)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current[j] = min(substitution, insertion, deletion)
        previous = current
    return previous[-1] / len(ref)
