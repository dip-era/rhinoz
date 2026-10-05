"""Evaluation metrics against a hand-written answer key.

- WER (raw vs refined), after normalising case, punctuation and spelled-out numbers
- term error rate on glossary terms (raw vs refined)
- protected-token violations: numbers/negations/modals/names the raw transcript got
  right but the refined transcript got wrong (target 0)
- decision precision/recall + "proposal leaks" (a rejected/deferred proposal reported as a decision)
- action-item precision/recall + invented owners/deadlines (target 0)

Item matching uses fuzzy text similarity - a proxy. Hand-check the matches it prints.
"""
from __future__ import annotations

import difflib
import re

from rapidfuzz import fuzz

from meeting_assistant.guards import MODALS, NEGATIONS, parse_numbers


def normalize_tokens(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("%", " percent ").replace("/", " ").replace("-", " ")
    t = re.sub(r"[^\w\s'.]", " ", t)
    toks = [x.strip(".'") for x in t.split()]
    toks = [x for x in toks if x]
    return [v if is_num else v.lower() for is_num, v in parse_numbers(toks)]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_tokens(ref), normalize_tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / len(r)


def _count(term: str, text: str) -> int:
    pat = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return len(re.findall(pat, text.lower()))


def term_error(ref: str, hyp: str, terms: list[str]) -> dict:
    per, ref_total, missed, inserted = {}, 0, 0, 0
    for t in terms:
        rc, hc = _count(t, ref), _count(t, hyp)
        per[t] = {"ref": rc, "hyp": hc}
        ref_total += rc
        missed += max(rc - hc, 0)
        inserted += max(hc - rc, 0)
    return {"term_error_rate": missed / ref_total if ref_total else 0.0, "missed": missed,
            "inserted": inserted, "ref_occurrences": ref_total, "per_term": per}


def _is_protected(tok: str, names: set[str]) -> bool:
    return (tok in NEGATIONS or tok.endswith("n't") or tok in MODALS or tok.endswith("'ll")
            or bool(re.fullmatch(r"\d+(\.\d+)?", tok)) or tok in names)


def _matched_ref_positions(ref: list[str], hyp: list[str]) -> set[int]:
    ok = set()
    for op, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if op == "equal":
            ok.update(range(i1, i2))
    return ok


def protected_violations(ref: str, raw: str, refined: str, names: list[str]) -> dict:
    r = normalize_tokens(ref)
    nameset = {n.lower() for n in names}
    prot = [i for i, t in enumerate(r) if _is_protected(t, nameset)]
    raw_ok = _matched_ref_positions(r, normalize_tokens(raw))
    ref_ok = _matched_ref_positions(r, normalize_tokens(refined))
    broken = [r[i] for i in prot if i in raw_ok and i not in ref_ok]
    fixed = [r[i] for i in prot if i not in raw_ok and i in ref_ok]
    return {"violations": len(broken), "broken_tokens": broken, "fixed_tokens": fixed, "protected_in_ref": len(prot)}


def match_items(pred: list[str], key: list[str], threshold: float = 55.0) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by token_set_ratio. Returns (pred_idx, key_idx, score)."""
    pairs = sorted(
        ((fuzz.token_set_ratio(p.lower(), k.lower()), i, j) for i, p in enumerate(pred) for j, k in enumerate(key)),
        reverse=True,
    )
    used_p, used_k, out = set(), set(), []
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in used_p or j in used_k:
            continue
        used_p.add(i)
        used_k.add(j)
        out.append((i, j, s))
    return out


def _pr(n_match: int, n_pred: int, n_key: int) -> tuple[float, float]:
    p = n_match / n_pred if n_pred else (1.0 if n_key == 0 else 0.0)
    r = n_match / n_key if n_key else 1.0
    return p, r


def decision_metrics(pred: list[str], key: list[str], not_decisions: list[str]) -> dict:
    m = match_items(pred, key)
    unmatched = [i for i in range(len(pred)) if i not in {a for a, _, _ in m}]
    leaks = match_items([pred[i] for i in unmatched], not_decisions)
    p, r = _pr(len(m), len(pred), len(key))
    return {"precision": p, "recall": r, "proposal_leaks": len(leaks),
            "matches": [(pred[i], key[j], round(s)) for i, j, s in m]}


def _norm_owner(s: str | None) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def action_metrics(pred: list[dict], key: list[dict]) -> dict:
    m = match_items([a["task"] for a in pred], [k["task"] for k in key])
    invented_owner, invented_deadline, missed_owner, details = 0, 0, 0, []
    for i, j, s in m:
        a, k = pred[i], key[j]
        po, ko = a.get("owner", "unspecified"), k.get("owner")
        if po != "unspecified":
            if a.get("owner_source") == "speaker_label":
                ok = k.get("owner_type") == "self"  # can't map labels to names; derived from audio, not invented
            else:
                ok = ko is not None and fuzz.ratio(_norm_owner(po), _norm_owner(ko)) >= 80
            if not ok:
                invented_owner += 1
                details.append(f"invented/wrong owner '{po}' for '{k['task']}' (key: {ko})")
        elif ko is not None:
            missed_owner += 1
        pd, kd = a.get("deadline", "unspecified"), k.get("deadline")
        if pd != "unspecified" and (kd is None or fuzz.token_set_ratio(pd.lower(), kd.lower()) < 70):
            invented_deadline += 1
            details.append(f"invented/wrong deadline '{pd}' for '{k['task']}' (key: {kd})")
    p, r = _pr(len(m), len(pred), len(key))
    return {"precision": p, "recall": r, "invented_owners": invented_owner, "invented_deadlines": invented_deadline,
            "missed_owners": missed_owner, "details": details,
            "matches": [(pred[i]["task"], key[j]["task"], round(s)) for i, j, s in m]}
