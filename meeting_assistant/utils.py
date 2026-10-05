"""Small shared helpers: text normalisation, fuzzy verbatim matching, chunking, GPU cleanup."""
from __future__ import annotations

import gc
import re
from typing import Callable, Sequence, TypeVar

from rapidfuzz import fuzz

T = TypeVar("T")

EDGE_PUNCT = " \t\n\"'“”‘’.,;:!?()[]{}"


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _straighten(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def norm_word(w: str) -> str:
    """Lowercase, drop punctuation (keeps inner apostrophes/underscores)."""
    w = _straighten(w.lower())
    w = re.sub(r"[^\w']+", "", w)
    return w.strip("'")


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_text(s: str) -> str:
    s = _straighten(s.lower())
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


def chunk_by_words(items: Sequence[T], max_words: int, text_of: Callable[[T], str] = lambda x: x.text) -> list[list[T]]:
    chunks: list[list[T]] = []
    cur: list[T] = []
    n = 0
    for it in items:
        w = len(text_of(it).split())
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += w
    if cur:
        chunks.append(cur)
    return chunks


def fuzzy_contains(needle: str, hay: str, threshold: float = 85.0) -> bool:
    n, h = norm_text(needle), norm_text(hay)
    if not n or not h:
        return False
    if n in h:
        return True
    return fuzz.partial_ratio(n, h) >= threshold


def find_verbatim(needle: str, hay: str, threshold: float = 85.0) -> str | None:
    """Return the substring of `hay` (original casing, word-aligned) that best matches `needle`.

    Used so that owners/deadlines in the record are copied from the transcript, never
    from the LLM's wording.
    """
    if not needle or not hay:
        return None
    nl, hl = _straighten(needle.lower()).strip(EDGE_PUNCT), _straighten(hay.lower())
    if not nl:
        return None
    idx = hl.find(nl)
    if idx >= 0:
        ds, de = idx, idx + len(nl)
    else:
        if len(nl) > len(hl):
            return None
        al = fuzz.partial_ratio_alignment(nl, hl)
        if al is None or al.score < threshold:
            return None
        ds, de = al.dest_start, al.dest_end
    while ds > 0 and hay[ds - 1].isalnum():
        ds -= 1
    while de < len(hay) and hay[de].isalnum():
        de += 1
    out = hay[ds:de].strip(EDGE_PUNCT)
    return out or None


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
