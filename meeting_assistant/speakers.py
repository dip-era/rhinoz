"""Speaker naming: map diarization labels (SPEAKER_00) to names, only when the meeting itself says them.

    turns -> chunks (< 2,500 words) -> LLM claims (self_intro / addressed, with quotes)
          -> deterministic validation against the transcript -> weighted vote per label

Weights: self_intro = 3 (high confidence), addressed = 1 (low confidence).
A claim survives only if:
  * the name is spoken (verbatim, fuzzy) in the cited segment and the quote is found there;
  * self_intro: the cited segment is spoken BY that label;
  * addressed: the cited segment is spoken by someone else AND that label answers within the next 2 turns.
Labels without surviving evidence keep their generic SPEAKER_xx name.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable

from . import prompts
from .config import Settings
from .llm_client import LLMClient
from .phonetics import term_match_score
from .schemas import SpeakerEvidence, SpeakerIdentity, SpeakerIdOut, Transcript
from .utils import alnum, chunk_by_words, find_verbatim, fuzzy_contains, norm_word

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 1 · Speaker names"
WEIGHTS = {"self_intro": 3, "addressed": 1}
_NOT_NAMES = {"i", "me", "you", "we", "everyone", "everybody", "guys", "team", "all", "folks", "sir", "madam", "speaker"}


def _key(name: str) -> str:
    return re.sub(r"'s$", "", norm_word(name.replace(" ", "")))


def _match_label(raw: str, labels: list[str]) -> str | None:
    raw = raw.strip()
    if raw in labels:
        return raw
    digits = re.findall(r"\d+", raw)  # "SPEAKER_1" / "Speaker 1" -> SPEAKER_01
    if digits:
        for lab in labels:
            d = re.findall(r"\d+", lab)
            if d and int(d[-1]) == int(digits[-1]):
                return lab
    return None


def display(name: str | None, role: str | None, label: str) -> str:
    if not name:
        return label
    if role and role.islower():
        role = role.title()  # "project manager" -> "Project Manager"
    return f"{name} ({role})" if role else name


def identity_map(speakers: list[SpeakerIdentity]) -> dict[str, SpeakerIdentity]:
    return {s.label: s for s in speakers}


def name_map(speakers: list[SpeakerIdentity]) -> dict[str, str]:
    return {s.label: s.display_name for s in speakers}


def resolve_speakers(
    transcript: Transcript,
    llm: LLMClient | None,
    settings: Settings,
    attendees: list[str] | None = None,
    progress: Progress = lambda *a: None,
) -> tuple[list[SpeakerIdentity], list[str]]:
    segs = transcript.segments
    labels = sorted({s.speaker for s in segs if s.speaker})
    warnings: list[str] = []
    if not labels:
        return [], warnings
    if llm is None:
        return [SpeakerIdentity(label=l, display_name=l) for l in labels], warnings

    segmap = {s.id: s for s in segs}
    order = {s.id: i for i, s in enumerate(segs)}
    thr = settings.quote_match_threshold
    evidence: dict[str, list[SpeakerEvidence]] = defaultdict(list)
    seen: set[tuple] = set()

    def validate(c) -> tuple[SpeakerEvidence | None, str, str | None]:
        kind = c.kind.strip().lower().replace("-", "_").replace(" ", "_")
        if kind not in WEIGHTS:
            return None, f"unknown kind '{c.kind}'", None
        label = _match_label(c.speaker, labels)
        if label is None:
            return None, f"unknown speaker label '{c.speaker}'", None
        seg = segmap.get(c.segment_id.strip())
        if seg is None:
            return None, f"unknown segment '{c.segment_id}'", label
        name = find_verbatim(c.name, seg.text, 90.0)
        if not name:
            return None, f"name not spoken in {seg.id}", label
        name = re.sub(r"['’]s$", "", name).strip()
        if len(name) < 2 or not name[0].isalpha() or _key(name) in _NOT_NAMES:
            return None, f"'{name}' is not a usable name", label
        if c.quote and not fuzzy_contains(c.quote, seg.text, thr):
            return None, f"quote not found in {seg.id}", label
        if kind == "self_intro" and seg.speaker != label:
            return None, f"{seg.id} is spoken by {seg.speaker}, not {label}", label
        if kind == "addressed":
            if seg.speaker == label:
                return None, f"{label} cannot address themself", label
            i = order[seg.id]
            if not any(s.speaker == label for s in segs[i + 1 : i + 3]):
                return None, f"{label} does not answer right after {seg.id}", label
        role = find_verbatim(c.role, seg.text, 90.0) if c.role else None
        ev = SpeakerEvidence(kind=kind, name=name, role=role, segment_id=seg.id, quote=c.quote or seg.text[:120],
                             weight=WEIGHTS[kind])
        return ev, "", label

    chunks = chunk_by_words(segs, settings.speaker_id_chunk_words)
    for ci, chunk in enumerate(chunks, 1):
        progress(STAGE, f"Looking for names in the conversation (chunk {ci}/{len(chunks)})", ci / len(chunks))
        last = order[chunk[-1].id]
        lookahead = segs[last + 1 : last + 3]  # so "Bob, ...?" at a chunk edge can see Bob's reply
        lines = [f"[{s.id} | {s.speaker or '?'}] {s.text}" for s in chunk + lookahead]
        user = f"SPEAKER LABELS: {', '.join(labels)}\n\nTRANSCRIPT:\n" + "\n".join(lines)
        out = llm.call_json(prompts.SPEAKER_ID_SYSTEM, user, SpeakerIdOut, max_tokens=2000)
        for c in out.claims:
            ev, why, label = validate(c)
            if ev is None:
                warnings.append(f"Speaker naming: dropped claim '{c.name}' for {c.speaker} ({why})")
                continue
            k = (label, ev.segment_id, ev.kind, _key(ev.name))
            if k not in seen:
                seen.add(k)
                evidence[label].append(ev)

    # ---- weighted vote per label ----------------------------------------
    identities: list[SpeakerIdentity] = []
    for label in labels:
        evs = evidence.get(label, [])
        ident = SpeakerIdentity(label=label, display_name=label, evidence=evs)
        scores, spelled = Counter(), {}
        for e in evs:
            scores[_key(e.name)] += e.weight
            spelled.setdefault(_key(e.name), e.name)
        ranked = scores.most_common()
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            k, sc = ranked[0]
            win = [e for e in evs if _key(e.name) == k]
            roles = Counter(e.role for e in win if e.role)
            ident.name, ident.score = spelled[k], sc
            ident.role = roles.most_common(1)[0][0] if roles else None
            ident.confidence = "high" if any(e.kind == "self_intro" for e in win) else "low"
        elif ranked:
            ident.notes.append(f"tie between {', '.join(spelled[k] for k, _ in ranked[:2])} - left unnamed")
        identities.append(ident)

    # ---- one name cannot belong to two labels ---------------------------
    by_name: dict[str, list[SpeakerIdentity]] = defaultdict(list)
    for i in identities:
        if i.name:
            by_name[_key(i.name)].append(i)
    for group in by_name.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda i: -i.score)
        keep = group[0] if group[0].score > group[1].score else None
        for i in group:
            if i is not keep:
                i.notes.append(f"name '{i.name}' also claimed by another label with equal/higher evidence - left unnamed")
                i.name, i.role, i.confidence, i.score = None, None, "none", 0

    # ---- use the attendee-list spelling when it clearly matches ---------
    for i in identities:
        if i.name and attendees:
            best = max(attendees, key=lambda a: term_match_score(i.name, a))
            if alnum(best) != alnum(i.name) and term_match_score(i.name, best) >= settings.phonetic_threshold:
                i.notes.append(f"transcribed as '{i.name}', spelled as in the attendee list")
                i.name = best
        i.display_name = display(i.name, i.role, i.label)
    named = sum(1 for i in identities if i.name)
    progress(STAGE, f"{named} of {len(labels)} speakers named from the conversation", 1.0)
    return identities, warnings
