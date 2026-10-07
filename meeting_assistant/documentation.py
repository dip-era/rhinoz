"""Stage 3 - documentation with LLM #2 (a different model from LLM #1).

3a. extract_lifecycle: one LLM call per chunk tags speech acts and emits proposal /
    task events. A running open-state (proposals + tasks) is carried across chunks, so
    a proposal at minute 5 can be resolved at minute 40. Every event must cite a
    segment in the current chunk with a verbatim quote, or it is dropped.
3b. summarize: concise summary + organised minutes, written AFTER verification so
    the prose is consistent with the verified decisions/action items.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .config import Settings
from .llm_client import LLMClient
from .schemas import (
    DeadlineEvidence,
    DocChunkOut,
    LifecycleEvent,
    MinutePoint,
    MinutesSection,
    NameHint,
    NotesOut,
    OwnerEvidence,
    Proposal,
    RefinedSegment,
    SummaryOut,
    Task,
)
from rapidfuzz import fuzz

from .utils import chunk_by_words, fmt_ts, fuzzy_contains

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 3 · Documentation"

VALID_ACTS = {"proposal", "agreement", "objection", "decision", "commitment", "assignment", "question", "info"}
ACCEPT_ACTS = {"agreement", "decision", "commitment"}
REVIEW_ACTS = {"proposal", "decision", "assignment", "commitment"}  # acts that should always have an event behind them
PROPOSAL_STATUSES = {"accepted", "rejected", "deferred", "unresolved"}
TASK_KINDS = {"self_commitment", "assignment", "open_task", "request"}
TASK_EVENTS = {"confirmed", "declined", "owner", "deadline"}


@dataclass
class LifecycleResult:
    speech_acts: dict[str, list[str]]
    proposals: list[Proposal]
    tasks: list[Task]
    warnings: list[str] = field(default_factory=list)


def seg_line(seg: RefinedSegment, diarized: bool, names: dict[str, str] | None = None) -> str:
    spk = ""
    if diarized and seg.speaker:
        nm = (names or {}).get(seg.speaker)
        spk = f" | {nm} [{seg.speaker}]" if nm and nm != seg.speaker else f" | {seg.speaker}"
    return f"[{seg.id} | {fmt_ts(seg.start)}{spk}] {seg.text}"


def _open_state(proposals: dict[str, Proposal], tasks: dict[str, Task]) -> str:
    ops = [
        {"id": p.proposal_id, "description": p.description, "status": p.status, "raised_in": p.segment_id}
        for p in proposals.values()
        if p.status in ("proposed", "deferred")
    ]
    ots = [
        {"id": t.task_id, "description": t.description, "kind": t.kind, "status": t.status}
        for t in tasks.values()
        if t.status in ("open", "confirmed")
    ][-20:]
    return json.dumps({"open_proposals": ops, "tasks": ots}, ensure_ascii=False, indent=1)


def extract_lifecycle(
    segments: list[RefinedSegment],
    diarized: bool,
    llm: LLMClient,
    settings: Settings,
    progress: Progress = lambda *a: None,
    names: dict[str, str] | None = None,
) -> LifecycleResult:
    segmap = {s.id: s for s in segments}
    order = {s.id: i for i, s in enumerate(segments)}
    proposals: dict[str, Proposal] = {}
    tasks: dict[str, Task] = {}
    acts: dict[str, list[str]] = {}
    warnings: list[str] = []
    thr = settings.quote_match_threshold

    def quote_ok(seg_id: str, quote: str) -> bool:
        if not quote.strip():
            return False
        i = order[seg_id]
        if fuzzy_contains(quote, segmap[seg_id].text, thr):
            return True
        # diarized segments are short fragments: allow a quote to run into the neighbouring segments
        window = " ".join(s.text for s in segments[max(0, i - 2) : i + 3])
        return fuzzy_contains(quote, window, thr)

    def event(status: str, seg: RefinedSegment, quote: str) -> LifecycleEvent:
        return LifecycleEvent(status=status, segment_id=seg.id, quote=quote, start=seg.start, speaker=seg.speaker)

    def add_evidence(t: Task, item, seg: RefinedSegment, chunk_ids: set[str]) -> None:
        if item.owner_text or item.owner_is_speaker:
            t.owner_evidence.append(
                OwnerEvidence(owner_text=item.owner_text, owner_is_speaker=item.owner_is_speaker, segment_id=seg.id, quote=item.quote)
            )
        if item.deadline_text:
            t.deadline_evidence.append(DeadlineEvidence(deadline_text=item.deadline_text, segment_id=seg.id, quote=item.quote))
        if item.context_name:
            q = item.context_quote or item.quote
            hit = next((sid for sid in sorted(chunk_ids, key=order.get) if fuzzy_contains(q, segmap[sid].text, thr)), None)
            if hit:
                t.name_hints.append(NameHint(name=item.context_name, segment_id=hit, quote=q))

    chunks = chunk_by_words(segments, settings.doc_chunk_words)
    prev_tail: list[RefinedSegment] = []
    for ci, chunk in enumerate(chunks, 1):
        progress(STAGE, f"LLM #2 tagging speech acts & lifecycle (chunk {ci}/{len(chunks)})", ci / len(chunks))
        chunk_ids = {s.id for s in chunk}
        user = (
            f"OPEN STATE:\n{_open_state(proposals, tasks)}\n\n"
            "CONTEXT (read-only):\n" + ("\n".join(seg_line(s, diarized, names) for s in prev_tail) or "(start of meeting)")
            + "\n\nCHUNK:\n" + "\n".join(seg_line(s, diarized, names) for s in chunk)
        )
        out = llm.call_json(prompts.DOC_SYSTEM, user, DocChunkOut, max_tokens=4000)

        def grounded(item, what: str) -> bool:
            """The quote must exist in the transcript. If the LLM mis-copied the segment id, re-anchor the event
            to the CHUNK segment that really contains the quote (nearest to the cited one)."""
            seg_id, quote = item.segment_id.strip(), item.quote
            if seg_id in chunk_ids and quote_ok(seg_id, quote):
                return True
            i0 = order.get(seg_id)
            pool = sorted(chunk, key=lambda x: abs(order[x.id] - i0)) if i0 is not None else list(chunk)
            hit = next((x for x in pool if quote.strip() and fuzzy_contains(quote, x.text, thr)), None)
            if hit is None:
                hit = next((x for x in pool if quote_ok(x.id, quote)), None)
            if hit is not None:
                warnings.append(f"{what}: cited {seg_id} but the quote is in {hit.id} - re-anchored")
                item.segment_id = hit.id
                return True
            warnings.append(f"{what}: quote not found in {seg_id} or anywhere in this chunk ('{quote[:60]}') - dropped")
            return False

        def process(out: DocChunkOut, tag_acts: bool) -> None:
            if tag_acts:
                for sa in out.speech_acts:
                    if sa.segment_id in chunk_ids:
                        acts[sa.segment_id] = [a for a in (x.strip().lower() for x in sa.acts) if a in VALID_ACTS] or ["info"]
            for s in chunk:
                acts.setdefault(s.id, ["info"])

            # ---- decisions stated as settled (no separate proposal/agreement) --
            ref_map: dict[str, str] = {}
            for nd in out.new_decisions:
                if not grounded(nd, f"decision '{nd.description[:40]}'"):
                    continue
                seg = segmap[nd.segment_id]
                if nd.quote.strip().endswith("?") or acts.get(seg.id) == ["question"]:
                    warnings.append(f"decision '{nd.description[:40]}' at {seg.id} is a question, not a decision - dropped")
                    continue
                if any(abs(order[p.segment_id] - order[seg.id]) <= 8
                       and fuzz.token_set_ratio(p.description.lower(), nd.description.lower()) >= 85
                       for p in proposals.values()):
                    continue  # already recorded
                if "decision" not in acts[seg.id]:
                    acts[seg.id] = [a for a in acts[seg.id] if a != "info"] + ["decision"]
                pid = f"P{len(proposals) + 1}"
                if nd.ref:
                    ref_map[nd.ref] = pid
                proposals[pid] = Proposal(
                    proposal_id=pid, description=nd.description.strip(), segment_id=seg.id, quote=nd.quote,
                    proposed_by=seg.speaker if diarized else None, status="accepted", basis="announced",
                    history=[event("accepted", seg, nd.quote)],
                )

            # ---- proposals ----------------------------------------------------
            for np_ in out.new_proposals:
                if not grounded(np_, f"proposal '{np_.description[:40]}'"):
                    continue
                pid = f"P{len(proposals) + 1}"
                ref_map[np_.ref] = pid
                seg = segmap[np_.segment_id]
                proposals[pid] = Proposal(
                    proposal_id=pid, description=np_.description.strip(), segment_id=seg.id, quote=np_.quote,
                    proposed_by=seg.speaker if diarized else None, history=[event("proposed", seg, np_.quote)],
                )
            for u in sorted(out.proposal_updates, key=lambda u: order.get(u.segment_id, 10**9)):
                pid = ref_map.get(u.id, u.id)
                status = u.status.strip().lower()
                if pid not in proposals or status not in PROPOSAL_STATUSES:
                    warnings.append(f"proposal update {u.id}->{u.status}: unknown id or status - dropped")
                    continue
                if not grounded(u, f"{pid} -> {status}"):
                    continue
                p, seg = proposals[pid], segmap[u.segment_id]
                if order[seg.id] < order[p.segment_id]:
                    warnings.append(f"{pid} -> {status} at {seg.id} precedes the proposal itself - dropped")
                    continue
                if status == "accepted":
                    seg_acts = set(acts.get(seg.id, []))
                    if not seg_acts & ACCEPT_ACTS:
                        warnings.append(f"{pid} acceptance at {seg.id} ignored: segment is not an agreement/decision/commitment")
                        continue
                    if diarized and seg.speaker and seg.speaker == p.proposed_by and "decision" not in seg_acts:
                        warnings.append(f"{pid} acceptance at {seg.id} ignored: the proposer agreeing with themself")
                        continue
                p.history.append(event(status, seg, u.quote))
                p.status = status

            # ---- tasks --------------------------------------------------------
            tref: dict[str, str] = {}
            for nt in out.new_tasks:
                if not nt.is_followup:
                    warnings.append(f"'{nt.description[:50]}' happens during the meeting - not an action item")
                    continue
                if not grounded(nt, f"task '{nt.description[:40]}'"):
                    continue
                seg = segmap[nt.segment_id]
                dup = next((t for t in tasks.values() if abs(order[t.segment_id] - order[seg.id]) <= 8
                            and fuzz.token_set_ratio(t.description.lower(), nt.description.lower()) >= 85), None)
                if dup:  # same task restated a few lines later: merge the evidence instead of listing it twice
                    tref[nt.ref] = dup.task_id
                    dup.history.append(event(nt.kind.strip().lower(), seg, nt.quote))
                    add_evidence(dup, nt, seg, chunk_ids)
                    warnings.append(f"'{nt.description[:50]}' merged into {dup.task_id} (same task restated)")
                    continue
                tid = f"T{len(tasks) + 1}"
                tref[nt.ref] = tid
                kind = nt.kind.strip().lower() if nt.kind.strip().lower() in TASK_KINDS else "request"
                seg = segmap[nt.segment_id]
                lp = ref_map.get(nt.linked_proposal, nt.linked_proposal) if nt.linked_proposal else None
                t = Task(
                    task_id=tid, description=nt.description.strip(), kind=kind, segment_id=seg.id, quote=nt.quote,
                    status="open" if kind == "request" else "confirmed",
                    linked_proposal=lp if lp in proposals else None, history=[event(kind, seg, nt.quote)],
                )
                add_evidence(t, nt, seg, chunk_ids)
                tasks[tid] = t
            for tu in sorted(out.task_updates, key=lambda u: order.get(u.segment_id, 10**9)):
                tid = tref.get(tu.id, tu.id)
                ev = tu.event.strip().lower()
                if tid not in tasks or ev not in TASK_EVENTS:
                    warnings.append(f"task update {tu.id}:{tu.event}: unknown id or event - dropped")
                    continue
                if not grounded(tu, f"{tid}:{ev}"):
                    continue
                t, seg = tasks[tid], segmap[tu.segment_id]
                if ev == "confirmed":
                    t.status, t.explicit_confirmation = "confirmed", True
                elif ev == "declined":
                    t.status = "declined"
                t.history.append(event(ev, seg, tu.quote))
                add_evidence(t, tu, seg, chunk_ids)
        process(out, tag_acts=True)

        # ---- consistency pass: an act with no event behind it means LLM #2 skipped something ----
        cited = {e.segment_id for p in proposals.values() for e in p.history} | {
            e.segment_id for t in tasks.values() for e in t.history}
        missing = [
            s for s in chunk
            if set(acts.get(s.id, [])) & REVIEW_ACTS
            and not any(c in order and abs(order[c] - order[s.id]) <= 2 for c in cited)
        ]
        if missing:
            progress(STAGE, f"LLM #2 reviewing {len(missing)} uncovered segments (chunk {ci})", None)
            review = (  # fresh open state, so the review can accept proposals created in the first pass by id
                f"OPEN STATE:\n{_open_state(proposals, tasks)}\n\n"
                + user[user.index("CONTEXT (read-only):"):]
                + "\n\nREVIEW: you tagged these CHUNK segments as proposal/decision/assignment/commitment, but no "
                "decision, proposal, proposal update or task cites them:\n"
                + "\n".join(f"- {s.id} ({', '.join(acts[s.id])}): {s.text}" for s in missing)
                + "\nEmit the missing new_decisions / new_proposals / proposal_updates / new_tasks for them now (read "
                "the neighbouring segments - sentences run over several segments). Leave out any that are genuinely "
                "not a decision, proposal or follow-up task. Use the same JSON format; speech_acts may be empty."
            )
            process(llm.call_json(prompts.DOC_SYSTEM, review, DocChunkOut, max_tokens=3000), tag_acts=False)
        prev_tail = chunk[-4:]

    # ---- end of meeting ----------------------------------------------------
    for p in proposals.values():
        if p.status == "proposed":
            p.status = "unresolved"
            p.notes.append("no conclusion reached by the end of the meeting")
    for t in tasks.values():
        lp = proposals.get(t.linked_proposal or "")
        if lp and lp.status != "accepted" and t.status == "confirmed" and not t.explicit_confirmation:
            t.status = "open"
            t.notes.append(f"depends on {lp.proposal_id}, which was {lp.status}, not accepted")
    return LifecycleResult(acts, list(proposals.values()), list(tasks.values()), warnings)


def summarize(
    segments: list[RefinedSegment],
    diarized: bool,
    decisions: list,
    action_items: list,
    not_adopted: list,
    llm: LLMClient,
    settings: Settings,
    progress: Progress = lambda *a: None,
    names: dict[str, str] | None = None,
) -> tuple[str, list[MinutesSection], list[str]]:
    warnings: list[str] = []
    valid = {s.id for s in segments}
    total_words = sum(len(s.text.split()) for s in segments)
    if total_words <= settings.summary_max_words:
        body = "TRANSCRIPT:\n" + "\n".join(seg_line(s, diarized, names) for s in segments)
    else:  # map-reduce for long meetings (stays under free-tier request size limits)
        notes = []
        chunks = chunk_by_words(segments, settings.doc_chunk_words * 3)
        for ci, chunk in enumerate(chunks, 1):
            progress(STAGE, f"LLM #2 extracting notes (chunk {ci}/{len(chunks)})", ci / len(chunks))
            out = llm.call_json(prompts.NOTES_SYSTEM, "TRANSCRIPT:\n" + "\n".join(seg_line(s, diarized, names) for s in chunk), NotesOut)
            notes.extend(out.notes)
        body = "NOTES (extracted from the transcript, in order):\n" + "\n".join(
            f"- {n.text} [{', '.join(i for i in n.segment_ids if i in valid)}]" for n in notes
        )

    def fmt_list(items, fn) -> str:
        return "\n".join(fn(x) for x in items) or "(none)"

    lists = (
        "DECISIONS (verified):\n" + fmt_list(decisions, lambda d: f"- {d.id}: {d.decision}")
        + "\n\nACTION ITEMS (verified):\n"
        + fmt_list(action_items, lambda a: f"- {a.id}: {a.task} (owner: {a.owner}; deadline: {a.deadline})")
        + "\n\nPROPOSALS NOT ADOPTED:\n" + fmt_list(not_adopted, lambda p: f"- {p.id} ({p.status}): {p.proposal}")
    )
    progress(STAGE, "LLM #2 writing summary and minutes", None)
    out = llm.call_json(prompts.SUMMARY_SYSTEM, lists + "\n\n" + body, SummaryOut, max_tokens=3000)
    sections = []
    for sec in out.minutes:
        pts = []
        for p in sec.points:
            ids = [i for i in p.segment_ids if i in valid]
            if not ids:
                warnings.append(f"Minutes point without a valid citation: '{p.text[:60]}'")
            pts.append(MinutePoint(text=p.text, segment_ids=ids, cited=bool(ids)))
        sections.append(MinutesSection(topic=sec.topic, points=pts))
    return out.summary.strip(), sections, warnings
