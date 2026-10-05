"""Double-Metaphone matching of word n-grams against glossary terms.

Word boundaries are dropped before encoding, so a multi-word mishearing can match a
single term: "cube earnest" -> KPRNST, "Kubernetes" -> KPRNTS (OSA similarity 0.83).
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz.distance import OSA, Levenshtein

try:
    from metaphone import doublemetaphone

    def _encode(s: str) -> tuple[str, ...]:
        return tuple(c for c in dict.fromkeys(doublemetaphone(s)) if c)

except ImportError:  # fallback: single metaphone
    import jellyfish

    def _encode(s: str) -> tuple[str, ...]:
        c = jellyfish.metaphone(s)
        return (c,) if c else ()


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


@lru_cache(maxsize=100_000)
def codes(text: str) -> tuple[str, ...]:
    lt = letters(text)
    return _encode(lt) if lt else ()


def phonetic_similarity(a: str, b: str) -> float:
    ca, cb = codes(a), codes(b)
    if not ca or not cb:
        return 0.0
    return max(OSA.normalized_similarity(x, y) for x in ca for y in cb)


@lru_cache(maxsize=200_000)
def term_match_score(heard: str, term: str) -> float:
    """How strongly `heard` (transcript words) sounds like glossary `term` (0..1)."""
    la, lb = letters(heard), letters(term)
    if not la or not lb:
        return 0.0
    if la == lb:
        return 1.0  # same letters, only formatting differs ("next js" vs "Next.js")
    ratio = len(la) / len(lb)
    if ratio < 0.5 or ratio > 2.0:
        return 0.0
    ph = phonetic_similarity(heard, term)
    tcodes = codes(term)
    if tcodes and min(len(c) for c in tcodes) < 3:
        # Short codes ("Jira" -> JR) match too much; demand an exact code and some spelling overlap.
        ortho = Levenshtein.normalized_similarity(la, lb)
        return ph if (ph == 1.0 and ortho >= 0.5) else 0.0
    return ph


def best_subphrase_score(text: str, term: str, max_n: int = 4) -> float:
    """Best term_match_score over contiguous sub-phrases of `text`."""
    toks = text.split()
    best = 0.0
    for i in range(len(toks)):
        for j in range(i + 1, min(len(toks), i + max_n) + 1):
            best = max(best, term_match_score(" ".join(toks[i:j]), term))
    return best
