"""Protected-token guard: numbers, negations, modal/commitment words and names.

An edit that changes any of these is marked `strict` and needs strong acoustic
evidence (or is rejected when no acoustic check is available).
"""
from __future__ import annotations

import re
from collections import Counter

from .utils import norm_word

_EDGE = " \"'“”‘’.,;:!?()[]{}"

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "cannot", "without", "nope"}
MODALS = {"will", "would", "shall", "should", "must", "can", "could", "may", "might"}
_NT_STEMS = {"won": "will", "can": "can", "shan": "shall", "wouldn": "would", "shouldn": "should",
             "couldn": "could", "mustn": "must", "mightn": "might"}

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
ORDINALS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
ORDINALS.update({"thirtieth": 30})
_NUM_RE = re.compile(r"^[$€£₹]?(\d+(?:,\d{3})*(?:\.\d+)?)(?:%|k|m|bn|st|nd|rd|th|s|ms)?$")

STRICT_KINDS = {"negation", "number", "modal"}


def _clean(tok: str) -> str:
    return tok.lower().replace("’", "'").strip(_EDGE)


def negation_count(tokens: list[str]) -> int:
    return sum(1 for t in map(_clean, tokens) if t in NEGATIONS or t.endswith("n't"))


def modal_list(tokens: list[str]) -> list[str]:
    out = []
    for t in map(_clean, tokens):
        if t in MODALS:
            out.append(t)
        elif t == "cannot":
            out.append("can")
        elif t.endswith("n't") and _NT_STEMS.get(t[:-3], t[:-3]) in MODALS:
            out.append(_NT_STEMS.get(t[:-3], t[:-3]))
        elif t.endswith("'ll"):
            out.append("will")
        elif t.endswith("'d"):
            out.append("would")
    return out


def _canon(num: str) -> str:
    num = num.replace(",", "")
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num or "0"


def parse_numbers(tokens: list[str]) -> list[tuple[bool, str]]:
    """Tokenise into (is_number, text); spelled-out numbers become canonical digits.

    "forty two" -> "42", "zero point eight one" -> "0.81", "1,000" -> "1000", "fifteenth" -> "15".
    """
    out: list[tuple[bool, str]] = []
    total = cur = 0
    active = False
    after_scale = False  # "two hundred AND five" continues a number; "two and three" does not
    pending_and = False
    decimals: list[str] | None = None

    def flush():
        nonlocal total, cur, active, decimals, after_scale, pending_and
        if active:
            val = str(total + cur)
            if decimals:
                val += "." + "".join(decimals)
            out.append((True, val))
        if pending_and:
            out.append((False, "and"))
        total = cur = 0
        active = after_scale = pending_and = False
        decimals = None

    for raw in tokens:
        t = _clean(raw)
        parts = t.split("-") if "-" in t and all(p in UNITS or p in TENS for p in t.split("-")) else [t]
        for p in parts:
            if decimals is not None:
                if p in UNITS and UNITS[p] < 10:
                    decimals.append(str(UNITS[p]))
                    continue
                flush()
            m = _NUM_RE.match(p)
            if m:
                flush()
                out.append((True, _canon(m.group(1))))
            elif p in UNITS or p in TENS:
                cur += UNITS.get(p, TENS.get(p, 0))
                active, after_scale, pending_and = True, False, False
            elif p in SCALES:
                if p == "hundred":
                    cur = max(cur, 1) * 100
                else:
                    total += max(cur, 1) * SCALES[p]
                    cur = 0
                active, after_scale, pending_and = True, True, False
            elif p == "point" and active:
                decimals = []
            elif p == "and" and active and after_scale:
                pending_and = True
            elif p in ORDINALS:
                flush()
                out.append((True, str(ORDINALS[p])))
            else:
                flush()
                if p:
                    out.append((False, raw))
    flush()
    return out


def extract_numbers(tokens: list[str]) -> list[str]:
    return [v for is_num, v in parse_numbers(tokens) if is_num]


def name_tokens(seg_tokens: list[str], ls: int, le: int, names: set[str], terms: set[str]) -> list[str]:
    """Tokens in [ls, le) that look like people's names."""
    out = []
    for i in range(ls, le):
        t = seg_tokens[i].strip(_EDGE)
        if not t:
            continue
        base = re.sub(r"'s$", "", norm_word(t))
        if base in names:
            out.append(t)
            continue
        if len(t) < 2 or not t[0].isupper() or base in {"i", "i'm", "i'll", "i've", "i'd"} or base in terms:
            continue
        if i == 0 or seg_tokens[i - 1].rstrip()[-1:] in ".?!":
            continue  # sentence-initial capital: ambiguous, not treated as a name
        if t.isupper() and len(t) <= 5:
            continue  # acronyms are terms
        out.append(t)
    return out


def protected_changes(seg_tokens: list[str], ls: int, le: int, replacement: str,
                      names: set[str], terms: set[str]) -> list[str]:
    """Which protected properties of seg_tokens[ls:le] would `replacement` change?"""
    orig = seg_tokens[ls:le]
    rep = replacement.split()
    hits = []
    if negation_count(orig) != negation_count(rep):
        hits.append("negation")
    if Counter(extract_numbers(orig)) != Counter(extract_numbers(rep)):
        hits.append("number")
    if Counter(modal_list(orig)) != Counter(modal_list(rep)):
        hits.append("modal")
    rep_norm = {re.sub(r"'s$", "", norm_word(t)) for t in rep}
    for n in name_tokens(seg_tokens, ls, le, names, terms):
        if re.sub(r"'s$", "", norm_word(n)) not in rep_norm:
            hits.append(f"name:{n}")
    orig_norm = {re.sub(r"'s$", "", norm_word(t)) for t in orig}
    for t in rep:
        b = re.sub(r"'s$", "", norm_word(t))
        if b in names and b not in orig_norm:
            hits.append(f"name:+{t.strip(_EDGE)}")
    return hits


def is_strict(hits: list[str], glossary_backed: bool) -> bool:
    if any(h in STRICT_KINDS for h in hits):
        return True
    # Name protection is waived when the replacement is a phonetically matching glossary
    # term/attendee - e.g. "Cube Earnest" (capitalised, looks like a name) -> "Kubernetes".
    return any(h.startswith("name:") for h in hits) and not glossary_backed
