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
from .utils import alnum, chunk_by_words, find_verbatim, fuzzy_contains, norm_word, spelled_words

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 1 · Speaker names"
WEIGHTS = {"self_intro": 3, "addressed": 1}  # name evidence
ROLE_WEIGHTS = {"self_intro": 3, "self_role": 3, "named_role": 1}  # role evidence
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


def is_self_intro(text: str, name: str) -> bool:
    """The name follows an introducing phrase ("I'm X", "my name is X", "this is X", "call me X", "X here").
    A name that is merely mentioned ("let's ask X") is not a self-introduction."""
    if not name.strip():
        return False
    t = text.lower().replace("’", "'")
    n = re.escape(name.lower().split()[0])
    lead = r"(?:i'm|i am|im|my name is|my name's|name's|name is|this is|call me|it's|it is)"
    return bool(re.search(rf"\b{lead}\s+(?:(?:actually|also|just|still|called)\s+)?{n}\b", t)
                or re.search(rf"\b{n}\s+here\b", t))


def is_self_role(text: str, role: str) -> bool:
    """The speaker states the role about themself ("I'm the X", "I'll be the X", "my role is X")."""
    if not role.strip():
        return False
    t = text.lower().replace("’", "'")
    r = re.escape(role.lower().split()[0])
    lead = r"(?:i'm|i am|im|i'll be|i will be|i'm going to be|my role is|my job is|i work as|i'm working as)"
    return bool(re.search(rf"\b{lead}\s+(?:\w+\s+){{0,5}}?{r}", t))


def usable_role(role: str | None) -> bool:
    """A role is a short noun phrase ("project manager"), not an activity ("to design a ...")."""
    if not role or not role.strip():
        return False
    words = role.strip().split()
    return len(words) <= 4 and words[0].lower() not in {"to", "do", "doing", "make", "making", "design", "designing",
                                                        "work", "working", "handle", "handling", "build", "building"}


def _canonical_names(names: list[str]) -> dict[str, str]:
    """Map each name key to the key of the longest name that contains all its words (a first name -> the full name)."""
    toks = {_key(n): {norm_word(w) for w in n.split()} for n in names}
    roots: list[str] = []
    canon: dict[str, str] = {}
    for k in sorted(toks, key=lambda k: -len(toks[k])):
        root = next((r for r in roots if toks[k] <= toks[r]), None)
        if root is None:
            roots.append(k)
        canon[k] = root or k
    return canon


# Unambiguous self-introductions, detected without the LLM (a backstop when it misses one)
_INTRO_RE = re.compile(r"(?i:\b(?:my name is|my name's|call me|i am called))\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})")
_NOT_NAME_WORDS = {"I", "I'm", "I'll", "I've", "I'd", "And", "But", "So", "The"}


def _pattern_intros(segs) -> list[tuple[str, str, str, str]]:
    """(label, name, segment_id, quote) for 'my name is X'-style statements, spoken by that segment's speaker."""
    found = []
    for s in segs:
        if not s.speaker:
            continue
        for m in _INTRO_RE.finditer(s.text):
            words = []
            for w in m.group(1).split():
                if w in _NOT_NAME_WORDS:
                    break
                words.append(w.strip(".,!?;:"))
            if words:
                found.append((s.speaker, " ".join(words), s.id, m.group(0)))
    return found


def display(name: str | None, role: str | None, label: str) -> str:
    if role and role.islower():
        role = role.title()  # "project manager" -> "Project Manager"
    shown = name or label
    return f"{shown} ({role})" if role else shown


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
    named_roles: list[SpeakerEvidence] = []
    seen: set[tuple] = set()

    def anchor(c, kind: str, label: str | None):
        """The segment that really contains the claimed evidence. LLMs sometimes mis-copy segment ids
        (S0008 -> S0018), so when the cited segment lacks the words, search the transcript - nearest first."""
        cited_id = c.segment_id.strip()
        needles = [x for x in ((c.name if kind != "self_role" else None),
                               (c.role if kind in ("self_role", "named_role") else None)) if x and x.strip()]

        def fits(s) -> bool:
            if not needles or not all(find_verbatim(n, s.text, 90.0) for n in needles):
                return False
            if c.quote and not fuzzy_contains(c.quote, s.text, thr):
                return False
            if kind in ("self_intro", "self_role") and label and s.speaker != label:
                return False
            if kind == "self_intro" and not is_self_intro(s.text, c.name or ""):
                return False
            return not (kind == "self_role" and not is_self_role(s.text, c.role or ""))

        cited = segmap.get(cited_id)
        if cited is not None and fits(cited):
            return cited
        i0 = order.get(cited_id)
        pool = sorted(segs, key=lambda s: abs(order[s.id] - i0)) if i0 is not None else segs
        found = next((s for s in pool if fits(s)), None)
        if found is not None:
            warnings.append(f"Speaker naming: '{c.name or c.role}' cited {cited_id} but is in {found.id} - re-anchored")
            return found
        return cited

    def validate(c) -> tuple[SpeakerEvidence | None, str, str | None]:
        kind = c.kind.strip().lower().replace("-", "_").replace(" ", "_")
        if kind not in WEIGHTS and kind not in ROLE_WEIGHTS:
            return None, f"unknown kind '{c.kind}'", None
        label = _match_label(c.speaker, labels)
        if label is None and kind != "named_role":
            return None, f"unknown speaker label '{c.speaker}'", None
        seg = anchor(c, kind, label)
        if seg is None:
            return None, f"evidence not found near '{c.segment_id}'", label
        if kind in ("self_role", "named_role"):
            role = find_verbatim(c.role, seg.text, 90.0) if c.role else None
            if not usable_role(role):
                return None, f"role not stated in {seg.id}", label
            if kind == "self_role" and seg.speaker != label:
                return None, f"{seg.id} is spoken by {seg.speaker}, not {label}", label
            if kind == "self_role" and not is_self_role(seg.text, role):
                return None, f"{seg.id} does not state the role in the first person", label
            name = ""
            if kind == "named_role":
                name = find_verbatim(c.name, seg.text, 90.0) or ""
                if len(name) < 2:
                    return None, f"named person not spoken in {seg.id}", label
            ev = SpeakerEvidence(kind=kind, name=name, role=role, segment_id=seg.id, quote=c.quote or seg.text[:120],
                                 weight=ROLE_WEIGHTS[kind])
            return ev, "", label
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
        if kind == "self_intro" and not is_self_intro(seg.text, name):
            return None, f"'{name}' is mentioned in {seg.id}, not introduced", label
        if kind == "addressed":
            if seg.speaker == label:
                return None, f"{label} cannot address themself", label
            i = order[seg.id]
            if not any(s.speaker == label for s in segs[i + 1 : i + 3]):
                return None, f"{label} does not answer right after {seg.id}", label
        role = find_verbatim(c.role, seg.text, 90.0) if c.role else None
        role = role if usable_role(role) else None
        ev = SpeakerEvidence(kind=kind, name=name, role=role, segment_id=seg.id, quote=c.quote or seg.text[:120],
                             weight=WEIGHTS[kind])
        return ev, "", label

    for label, name, seg_id, quote in _pattern_intros(segs):  # backstop, validated like an LLM claim
        k = (label, seg_id, "self_intro", _key(name))
        if k not in seen and len(name) >= 2:
            seen.add(k)
            evidence[label].append(SpeakerEvidence(kind="self_intro", name=name, segment_id=seg_id, quote=quote,
                                                   weight=WEIGHTS["self_intro"]))

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
                warnings.append(f"Speaker naming: dropped claim '{c.name or c.role}' for {c.speaker} ({why})")
                continue
            if ev.kind == "named_role":  # attached to whichever label ends up with that name (after the vote)
                named_roles.append(ev)
                continue
            k = (label, ev.segment_id, ev.kind, _key(ev.name))
            if k not in seen:
                seen.add(k)
                evidence[label].append(ev)
            elif ev.role:  # same claim already found (e.g. by the pattern backstop): keep the role it adds
                same = next(e for e in evidence[label] if (e.segment_id, e.kind, _key(e.name)) == k[1:])
                same.role = same.role or ev.role

    # ---- weighted vote per label ----------------------------------------
    identities: list[SpeakerIdentity] = []
    for label in labels:
        evs = evidence.get(label, [])
        ident = SpeakerIdentity(label=label, display_name=label, evidence=evs)
        name_evs = [e for e in evs if e.kind in WEIGHTS]  # role-only evidence does not vote on the name
        canon = _canonical_names([e.name for e in name_evs])  # a first name and the full name are one person
        scores, spelled = Counter(), {}
        for e in name_evs:
            k = canon[_key(e.name)]
            scores[k] += e.weight
            spelled.setdefault(k, max((x.name for x in name_evs if canon[_key(x.name)] == k), key=len))
        ranked = scores.most_common()
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            k, sc = ranked[0]
            win = [e for e in name_evs if canon[_key(e.name)] == k]
            ident.name, ident.score = spelled[k], sc
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

    # ---- roles: own statements (self_intro / self_role) outweigh what others say (named_role) ----
    by_key = {_key(i.name): i for i in identities if i.name}
    for ev in named_roles:
        target = by_key.get(_key(ev.name)) or next(
            (i for i in identities if i.name and term_match_score(i.name.split()[0], ev.name.split()[0]) >= 0.8), None)
        if target is not None:
            target.evidence.append(ev)
    for i in identities:
        votes: Counter = Counter()
        for e in i.evidence:
            own_name = {norm_word(w) for w in (i.name or "").split()}
            if e.role and e.kind in ROLE_WEIGHTS and (
                    e.kind != "self_intro" or not i.name or {norm_word(w) for w in e.name.split()} <= own_name):
                votes[e.role.lower()] += e.weight
        if votes:
            top = votes.most_common(1)[0][0]
            i.role = next(e.role for e in i.evidence if e.role and e.role.lower() == top)

    # ---- a speaker spelling a word letter by letter overrides how the name was transcribed ----
    for idx, seg in enumerate(segs):
        for word in spelled_words(seg.text):
            ident = next((i for i in identities if i.label == seg.speaker), None)
            if ident is None:
                continue
            asked = any("spell" in x.text.lower() and "name" in x.text.lower() and x.speaker != seg.speaker
                        for x in segs[max(0, idx - 2) : idx])
            first = ident.name.split()[0] if ident.name else None
            if first and alnum(first) != alnum(word) and term_match_score(first, word) >= 0.75:
                ident.notes.append(f"spelled '{word}' letter by letter in {seg.id}; transcribed as '{first}'")
                ident.name = " ".join([word] + ident.name.split()[1:])
            elif not ident.name and asked:  # asked to spell their name, then spelled it
                ident.name, ident.confidence, ident.score = word, "high", WEIGHTS["self_intro"]
                ident.notes.append(f"named from the spelling in {seg.id} (asked to spell their name)")
            else:
                continue
            ident.evidence.append(SpeakerEvidence(kind="spelled", name=word, segment_id=seg.id, quote=seg.text[:120],
                                                  weight=WEIGHTS["self_intro"]))

    # ---- use the attendee-list spelling when it clearly matches ---------
    for i in identities:
        if i.name and attendees and not any(e.kind == "spelled" for e in i.evidence):
            best = max(attendees, key=lambda a: term_match_score(i.name, a))
            if alnum(best) != alnum(i.name) and term_match_score(i.name, best) >= settings.phonetic_threshold:
                i.notes.append(f"transcribed as '{i.name}', spelled as in the attendee list")
                i.name = best
        i.display_name = display(i.name, i.role, i.label)
    named = sum(1 for i in identities if i.name)
    progress(STAGE, f"{named} of {len(labels)} speakers named from the conversation", 1.0)
    return identities, warnings
