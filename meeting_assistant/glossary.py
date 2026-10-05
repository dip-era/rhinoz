"""Glossary = user-supplied agenda/terms + attendee names + terms LLM #1 proposes from context.

LLM-proposed terms are kept only if they are grounded: the term appears in the
transcript, or one of its `heard_as` phrases appears verbatim in a cited segment.
"""
from __future__ import annotations

import re
from typing import Callable

from . import prompts
from .config import Settings
from .llm_client import LLMClient
from .schemas import GlossaryOut, GlossaryTerm, Transcript
from .utils import alnum, chunk_by_words, norm_text

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 2 · Glossary"


def parse_term_list(text: str | None) -> list[str]:
    if not text:
        return []
    seen, out = set(), []
    for t in re.split(r"[\n,;]+", text):
        t = t.strip().strip("-*• ").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _marked_line(seg, transcript: Transcript, low_conf: float) -> str:
    """Segment text with low-confidence word runs wrapped in {braces}."""
    out, run = [], []
    for w in transcript.segment_words(seg):
        if w.prob < low_conf:
            run.append(w.text)
        else:
            if run:
                out.append("{" + " ".join(run) + "}")
                run = []
            out.append(w.text)
    if run:
        out.append("{" + " ".join(run) + "}")
    return f"[{seg.id}] " + " ".join(out)


def build_glossary(
    transcript: Transcript,
    user_terms: list[str],
    attendees: list[str],
    llm: LLMClient | None,
    settings: Settings,
    progress: Progress = lambda *a: None,
) -> tuple[list[GlossaryTerm], list[str]]:
    warnings: list[str] = []
    glossary: dict[str, GlossaryTerm] = {}
    for t in user_terms:
        glossary.setdefault(t.lower(), GlossaryTerm(term=t, source="user"))
    for n in attendees:
        glossary.setdefault(n.lower(), GlossaryTerm(term=n, source="attendee", category="name"))

    if llm is None:
        return list(glossary.values()), warnings

    segmap = transcript.segment_map()
    full_norm = norm_text(" ".join(s.text for s in transcript.segments))
    full_alnum = alnum(full_norm)
    chunks = chunk_by_words(transcript.segments, settings.glossary_chunk_words)
    for ci, chunk in enumerate(chunks, 1):
        progress(STAGE, f"LLM #1 proposing domain terms (chunk {ci}/{len(chunks)})", ci / len(chunks))
        known = ", ".join(g.term for g in glossary.values() if g.category == "term") or "(none)"
        user = (
            f"Terms already known (keep their spelling): {known}\n\n"
            "TRANSCRIPT:\n" + "\n".join(_marked_line(s, transcript, settings.low_conf) for s in chunk)
        )
        out = llm.call_json(prompts.GLOSSARY_SYSTEM, user, GlossaryOut, max_tokens=3000)
        for t in out.terms:
            term = t.term.strip()
            if not term or len(term.split()) > 5 or term.lower() in glossary:
                continue
            seg_ids = [s for s in t.segment_ids if s in segmap]
            cited_text = norm_text(" ".join(segmap[s].text for s in seg_ids))
            heard = [h for h in t.heard_as if h.strip() and norm_text(h) in (cited_text or full_norm)]
            appears = alnum(term) and alnum(term) in full_alnum
            if not appears and not heard:
                warnings.append(f"Glossary: dropped ungrounded LLM term '{term}'")
                continue
            glossary[term.lower()] = GlossaryTerm(
                term=term, source="llm", heard_as=heard, segment_ids=seg_ids, rationale=t.rationale[:200]
            )
    return list(glossary.values()), warnings


def whisper_prompt(glossary: list[GlossaryTerm], max_terms: int = 60) -> str | None:
    """Glossary prompt used identically for both hypotheses in acoustic scoring."""
    terms = [g.term for g in glossary][:max_terms]
    return ("Glossary: " + ", ".join(terms) + ".") if terms else None
