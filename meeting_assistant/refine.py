"""Stage 2 - evidence-gated refinement (LLM #1 is a constrained surgeon).

    glossary -> candidate spans -> LLM #1 proposes structured edits
    -> locate each edit verbatim -> formatting / protected-token checks
    -> teacher-forced acoustic scoring -> deterministic verdict -> apply

`decide`, `resolve_overlaps` and `apply_edits` are pure functions so that
eval/tune_thresholds.py can re-run the verdicts offline with other thresholds.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import prompts
from .audio_io import SAMPLE_RATE
from .candidates import find_candidates
from .config import Settings
from .glossary import build_glossary, whisper_prompt
from .guards import is_strict, protected_changes
from .llm_client import LLMClient
from .phonetics import best_subphrase_score
from .schemas import (
    CandidateSpan,
    EditProposalBatch,
    EditVerdict,
    GlossaryTerm,
    ProposedEdit,
    RefinedSegment,
    Transcript,
    Word,
)
from .utils import alnum, chunk_by_words, norm_word

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 2 · Refinement"
_TRAIL = re.compile(r"[.,!?;:]+$")


@dataclass
class Thresholds:
    low_conf: float = 0.5
    phonetic_threshold: float = 0.8
    tau_conf: float = 6.0
    tau_glossary: float = 3.0
    strong_margin: float = 0.5
    no_acoustic_min_llm_conf: float = 0.75

    @classmethod
    def from_settings(cls, s: Settings) -> "Thresholds":
        return cls(s.low_conf, s.phonetic_threshold, s.tau_conf, s.tau_glossary, s.strong_margin, s.no_acoustic_min_llm_conf)


@dataclass
class RefinementResult:
    glossary: list[GlossaryTerm]
    candidates: list[CandidateSpan]
    verdicts: list[EditVerdict]
    refined_segments: list[RefinedSegment]
    acoustic_used: bool
    acoustic_model: str | None
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# LLM #1 edit proposal
# ============================================================================


def _candidate_line(c: CandidateSpan) -> str:
    bits = []
    if "low_confidence" in c.sources:
        bits.append(f"ASR confidence {c.mean_prob:.0%}")
    if c.phonetic_term:
        bits.append(f'sounds like glossary term "{c.phonetic_term}" ({c.phonetic_score:.2f})')
    return f'{c.span_id} | {c.segment_id} | "{c.text}" | ' + "; ".join(bits)


def propose_edits(
    transcript: Transcript,
    glossary: list[GlossaryTerm],
    candidates: list[CandidateSpan],
    llm: LLMClient,
    settings: Settings,
    progress: Progress,
) -> list[ProposedEdit]:
    by_seg: dict[str, list[CandidateSpan]] = defaultdict(list)
    for c in candidates:
        by_seg[c.segment_id].append(c)
    gl_lines = []
    for g in glossary:
        line = f"- {g.term}" + (" (person)" if g.category == "name" else "")
        if g.heard_as:
            line += f"  [possibly transcribed as: {', '.join(g.heard_as)}]"
        gl_lines.append(line)
    gl_text = "\n".join(gl_lines) or "(empty)"

    proposals: list[ProposedEdit] = []
    chunks = chunk_by_words(transcript.segments, settings.refine_chunk_words)
    for ci, chunk in enumerate(chunks, 1):
        progress(STAGE, f"LLM #1 proposing edits (chunk {ci}/{len(chunks)})", ci / len(chunks))
        ids = {s.id for s in chunk}
        cands = [c for s in chunk for c in by_seg.get(s.id, [])]
        seg_lines = []
        for s in chunk:
            spk = f" ({s.speaker})" if transcript.diarized and s.speaker else ""
            seg_lines.append(f"[{s.id}]{spk} {s.text}")
        user = (
            f"GLOSSARY:\n{gl_text}\n\n"
            "CANDIDATE SPANS (span_id | segment | text | evidence):\n"
            + ("\n".join(_candidate_line(c) for c in cands) or "(none)")
            + "\n\nTRANSCRIPT:\n"
            + "\n".join(seg_lines)
        )
        out = llm.call_json(prompts.REFINE_SYSTEM, user, EditProposalBatch, max_tokens=3000)
        proposals.extend(e for e in out.edits if e.segment_id in ids)
    return proposals


# ============================================================================
# Locating and pre-checking edits
# ============================================================================


def locate(original: str, words: list[Word], hint: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Find `original` as a contiguous run of segment words (local indices, end exclusive)."""
    target = [t for t in (norm_word(x) for x in original.split()) if t]
    if not target:
        return None
    seg = [norm_word(w.text) for w in words]
    n = len(target)
    matches = [(i, i + n) for i in range(len(seg) - n + 1) if seg[i : i + n] == target]
    if not matches:  # tokenisation differs ("real time" vs "real-time"): compare concatenations
        joined = "".join(target)
        for i in range(len(seg)):
            acc = ""
            for j in range(i, min(len(seg), i + n + 3)):
                acc += seg[j]
                if acc == joined:
                    matches.append((i, j + 1))
                    break
                if len(acc) >= len(joined):
                    break
    if not matches:
        return None
    if hint:
        overlapping = [m for m in matches if m[0] < hint[1] and hint[0] < m[1]]
        if overlapping:
            return overlapping[0]
        return min(matches, key=lambda m: abs(m[0] - hint[0]))
    return matches[0]


def splice(tokens: list[str], ls: int, le: int, replacement: str) -> list[str]:
    rep = replacement.strip()
    trailing = _TRAIL.search(tokens[le - 1]) if le > ls else None
    if trailing and not _TRAIL.search(rep):
        rep += trailing.group(0)  # keep the sentence punctuation the replaced words carried
    return tokens[:ls] + [rep] + tokens[le:]


def formatting_equivalent(a: str, b: str) -> bool:
    return bool(alnum(a)) and alnum(a) == alnum(b)


def _glossary_support(original: str, replacement: str, glossary: list[GlossaryTerm], thr: float):
    best = (False, None, None)
    rep_al, orig_al = alnum(replacement), alnum(original)
    for g in glossary:
        ta = alnum(g.term)
        if not ta or ta not in rep_al or ta in orig_al:
            continue
        s = best_subphrase_score(original, g.term)
        if s >= thr and (best[2] is None or s > best[2]):
            best = (True, g.term, s)
    return best


def prepare_verdicts(
    proposals: list[ProposedEdit],
    transcript: Transcript,
    candidates: list[CandidateSpan],
    glossary: list[GlossaryTerm],
    thr: Thresholds,
) -> list[EditVerdict]:
    segmap = transcript.segment_map()
    cand_by_id = {c.span_id: c for c in candidates}
    cand_by_seg: dict[str, list[CandidateSpan]] = defaultdict(list)
    for c in candidates:
        cand_by_seg[c.segment_id].append(c)
    names = {norm_word(p) for g in glossary if g.category == "name" for p in g.term.split()}
    terms = {norm_word(p) for g in glossary if g.category == "term" for p in g.term.split()}

    out: list[EditVerdict] = []
    seen: set[tuple] = set()
    for k, p in enumerate(proposals, 1):
        v = EditVerdict(
            edit_id=f"E{k:04d}", span_id=p.span_id, segment_id=p.segment_id, original=p.original,
            replacement=p.replacement.strip(), edit_type=p.edit_type, claimed_edit_type=p.edit_type,
            reason=p.reason, confidence=p.confidence,
        )
        out.append(v)
        seg = segmap.get(p.segment_id)
        if seg is None:
            v.precheck_failed = "unknown segment id"
            continue
        words = transcript.segment_words(seg)
        base = seg.word_ids[0]
        hint = None
        c = cand_by_id.get(p.span_id or "")
        if c and c.segment_id == seg.id:
            hint = (c.word_start - base, c.word_end - base)
        loc = locate(p.original, words, hint)
        if loc is None:
            v.precheck_failed = "original text not found verbatim in the segment (possible hallucination)"
            continue
        ls, le = loc
        v.word_start, v.word_end = words[ls].id, words[le - 1].id + 1
        tokens = [w.text for w in words]
        v.located_text = " ".join(tokens[ls:le])
        v.mean_prob = float(np.mean([w.prob for w in words[ls:le]]))

        key = (v.word_start, v.word_end, v.replacement)
        if key in seen:
            v.precheck_failed = "duplicate of an earlier edit"
            continue
        seen.add(key)
        if not v.replacement:
            v.precheck_failed = "deletions are not allowed"
            continue
        if v.located_text.strip() == v.replacement:
            v.precheck_failed = "no-op edit"
            continue
        if len(v.replacement.split()) > 2 * (le - ls) + 3:
            v.precheck_failed = "replacement much longer than the original (rewrite, not a correction)"
            continue

        overl = [c for c in cand_by_seg.get(seg.id, []) if c.word_start < v.word_end and v.word_start < c.word_end]
        srcs = sorted({s for c in overl for s in c.sources}) if overl else ["llm_flag"]
        v.sources = srcs

        if v.edit_type == "formatting" and not formatting_equivalent(v.located_text, v.replacement):
            v.edit_type = "acoustic"  # the words differ when spoken: it must pass the acoustic check

        v.glossary_backed, v.glossary_term, v.phonetic_score = _glossary_support(
            v.located_text, v.replacement, glossary, thr.phonetic_threshold
        )
        v.protected_hits = protected_changes(tokens, ls, le, v.replacement, names, terms)
        v.strict = is_strict(v.protected_hits, v.glossary_backed)
    return out


# ============================================================================
# Acoustic scoring
# ============================================================================


def score_edits(
    verdicts: list[EditVerdict], transcript: Transcript, audio: np.ndarray, prompt: str | None,
    model_id: str, progress: Progress,
) -> None:
    from .acoustic import AcousticScorer

    todo = [v for v in verdicts if v.precheck_failed is None and v.edit_type == "acoustic"]
    if not todo:
        return
    progress(STAGE, f"Loading acoustic verifier {model_id}", None)
    scorer = AcousticScorer(model_id)
    try:
        segmap = transcript.segment_map()
        groups: dict[str, list[EditVerdict]] = defaultdict(list)
        for v in todo:
            groups[v.segment_id].append(v)
        for gi, (sid, vs) in enumerate(groups.items(), 1):
            progress(STAGE, f"Acoustic check: segment {gi}/{len(groups)}", gi / len(groups))
            seg = segmap[sid]
            s = max(0.0, seg.start - 0.25)
            e = min(transcript.duration, seg.end + 0.25, s + 30.0)
            enc = scorer.encode(audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)])
            tokens = [w.text for w in transcript.segment_words(seg)]
            base = seg.word_ids[0]
            lp0 = scorer.logprob(enc, " ".join(tokens), prompt)
            for v in vs:
                new_tokens = splice(tokens, v.word_start - base, v.word_end - base, v.replacement)
                lp1 = scorer.logprob(enc, " ".join(new_tokens), prompt)
                v.acoustic_logp_original, v.acoustic_logp_edit = lp0, lp1
                v.acoustic_delta = lp1 - lp0
    finally:
        scorer.close()


# ============================================================================
# Deterministic verdicts (pure)
# ============================================================================


def allowed_drop(v: EditVerdict, thr: Thresholds) -> float:
    """Low ASR confidence and glossary/phonetic support buy tolerance for a log-lik drop
    (Whisper's language-model prior penalises rare terms)."""
    mp = v.mean_prob if v.mean_prob is not None else 1.0
    return thr.tau_conf * (1.0 - mp) + (thr.tau_glossary if v.glossary_backed else 0.0)


def decide(v: EditVerdict, thr: Thresholds, acoustic_available: bool) -> tuple[bool, str]:
    if v.precheck_failed:
        return False, v.precheck_failed
    if v.edit_type == "formatting":
        return True, "formatting-only (same spoken words); acoustic check skipped"
    if acoustic_available and v.acoustic_delta is not None:
        d = v.acoustic_delta
        if v.strict:
            ok = d >= thr.strong_margin
            return ok, f"protected change {v.protected_hits}: needs Δlogp ≥ +{thr.strong_margin:.2f}, got {d:+.2f}"
        allow = allowed_drop(v, thr)
        v.acoustic_allowed_drop = allow
        return d >= -allow, f"Δlogp {d:+.2f} vs allowed drop {allow:.2f}"
    # ---- no acoustic evidence available: conservative fallback rules ----
    if v.strict:
        return False, f"protected change {v.protected_hits} without acoustic evidence"
    if v.glossary_backed and v.confidence >= 0.6:
        return True, f"no acoustic check; phonetic match to glossary term '{v.glossary_term}' ({v.phonetic_score:.2f})"
    if (v.mean_prob if v.mean_prob is not None else 1.0) < thr.low_conf and v.confidence >= thr.no_acoustic_min_llm_conf:
        return True, "no acoustic check; low ASR confidence and high LLM confidence"
    return False, "insufficient evidence without acoustic check"


def resolve_overlaps(verdicts: list[EditVerdict]) -> None:
    kept: list[EditVerdict] = []
    ranked = sorted(
        (v for v in verdicts if v.accepted),
        key=lambda v: (v.acoustic_delta if v.acoustic_delta is not None else 0.0, v.confidence),
        reverse=True,
    )
    for v in ranked:
        clash = next((k for k in kept if k.word_start < v.word_end and v.word_start < k.word_end), None)
        if clash:
            v.accepted = False
            v.verdict_reason = f"overlaps accepted edit {clash.edit_id}"
        else:
            kept.append(v)


def apply_edits(transcript: Transcript, verdicts: list[EditVerdict]) -> list[RefinedSegment]:
    by_seg: dict[str, list[EditVerdict]] = defaultdict(list)
    for v in verdicts:
        if v.accepted:
            by_seg[v.segment_id].append(v)
    out = []
    for seg in transcript.segments:
        tokens = [transcript.words[i].text for i in seg.word_ids]
        base = seg.word_ids[0]
        edits = sorted(by_seg.get(seg.id, []), key=lambda v: v.word_start, reverse=True)
        for v in edits:  # right-to-left keeps indices valid
            tokens = splice(tokens, v.word_start - base, v.word_end - base, v.replacement)
        out.append(
            RefinedSegment(
                id=seg.id, start=seg.start, end=seg.end, speaker=seg.speaker, text=" ".join(tokens),
                raw_text=seg.text, edit_ids=[v.edit_id for v in reversed(edits)],
            )
        )
    return out


def finalize(verdicts: list[EditVerdict], thr: Thresholds, acoustic_available: bool) -> None:
    for v in verdicts:
        v.accepted, v.verdict_reason = decide(v, thr, acoustic_available)
    resolve_overlaps(verdicts)


# ============================================================================
# Orchestration
# ============================================================================


def refine(
    transcript: Transcript,
    audio: np.ndarray,
    user_terms: list[str],
    attendees: list[str],
    settings: Settings,
    llm: LLMClient,
    progress: Progress = lambda *a: None,
) -> RefinementResult:
    thr = Thresholds.from_settings(settings)
    glossary, warnings = build_glossary(transcript, user_terms, attendees, llm, settings, progress)
    progress(STAGE, f"Glossary has {len(glossary)} terms; finding candidate spans", None)
    candidates = find_candidates(transcript, glossary, settings)
    progress(STAGE, f"{len(candidates)} candidate spans", None)
    proposals = propose_edits(transcript, glossary, candidates, llm, settings, progress)
    verdicts = prepare_verdicts(proposals, transcript, candidates, glossary, thr)

    acoustic_used, model_id = False, None
    needs = [v for v in verdicts if v.precheck_failed is None and v.edit_type == "acoustic"]
    if settings.acoustic_check and needs:
        model_id = settings.acoustic_model_id
        if not model_id:
            warnings.append(f"No HF checkpoint known for ASR model '{settings.asr_model}'; set ACOUSTIC_MODEL.")
        else:
            try:
                score_edits(verdicts, transcript, audio, whisper_prompt(glossary), model_id, progress)
                acoustic_used = True
            except Exception as e:  # OOM, missing transformers, download failure ...
                warnings.append(f"Acoustic verification unavailable ({type(e).__name__}: {e}); used fallback evidence rules.")
                model_id = None
    finalize(verdicts, thr, acoustic_used)
    refined = apply_edits(transcript, verdicts)
    n_acc = sum(v.accepted for v in verdicts)
    progress(STAGE, f"{n_acc}/{len(verdicts)} proposed edits accepted", 1.0)
    return RefinementResult(glossary, candidates, verdicts, refined, acoustic_used, model_id, warnings)
