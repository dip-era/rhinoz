"""Stage 4 - deterministic verification (+ optional NLI flags).

* Decisions = proposals whose final status is "accepted". Nothing else.
* Owner: either a name found verbatim (fuzzy) inside a cited segment, or the speaker
  label of a first-person commitment (diarization only). Otherwise "unspecified".
  Context-inferred names ("Thanks, Priya") go to `owner_annotation`, never `owner`.
* Deadline: copied verbatim from the cited segment, never normalised to a date.
* NLI (DeBERTa-MNLI) only ADDS a flag; it never deletes an item.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import Settings
from .documentation import LifecycleResult
from .schemas import ActionItem, Decision, Proposal, ProposalSummary, Provenance, RefinedSegment, Task
from .utils import find_verbatim, fmt_ts

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 4 · Verification"


@dataclass
class VerifiedOutputs:
    decisions: list[Decision]
    action_items: list[ActionItem]
    unconfirmed_requests: list[ActionItem]
    rejected: list[ProposalSummary]
    deferred: list[ProposalSummary]
    unresolved: list[ProposalSummary]
    nli_model: str | None = None
    warnings: list[str] = field(default_factory=list)


def _provenance(seg_ids: list[str], quotes: list[str], segmap: dict[str, RefinedSegment], order: dict[str, int]) -> Provenance:
    ids = sorted({s for s in seg_ids if s in segmap}, key=order.get)
    return Provenance(
        segment_ids=ids,
        start=min(segmap[s].start for s in ids) if ids else 0.0,
        end=max(segmap[s].end for s in ids) if ids else 0.0,
        quotes=list(dict.fromkeys(q for q in quotes if q)),
    )


def _proposal_summary(p: Proposal, segmap, order) -> ProposalSummary:
    return ProposalSummary(
        id=p.proposal_id, proposal=p.description, status=p.status, proposed_by=p.proposed_by,
        provenance=_provenance([e.segment_id for e in p.history], [e.quote for e in p.history], segmap, order),
    )


def _verify_task(t: Task, idx: int, segmap, order, diarized: bool, thr: float) -> ActionItem:
    notes: list[str] = []
    owner, owner_source, annotation = "unspecified", "unspecified", None

    for ev in reversed(t.owner_evidence):  # latest statement wins
        seg = segmap.get(ev.segment_id)
        if seg is None:
            continue
        if ev.owner_text:
            found = find_verbatim(ev.owner_text, seg.text, thr)
            if found:
                owner, owner_source = found, "stated"
                break
            notes.append(f"owner '{ev.owner_text}' not found in {ev.segment_id} - discarded")
        elif ev.owner_is_speaker:
            if diarized and seg.speaker:
                owner, owner_source = seg.speaker, "speaker_label"
                break
            notes.append(f"self-commitment in {ev.segment_id}, but speaker unknown (diarization off)")
            annotation = f"Self-commitment by the speaker of {ev.segment_id} ({fmt_ts(seg.start)}); speaker not identified."

    deadline = "unspecified"
    for ev in reversed(t.deadline_evidence):
        seg = segmap.get(ev.segment_id)
        found = find_verbatim(ev.deadline_text, seg.text, thr) if seg else None
        if found:
            deadline = found
            break
        notes.append(f"deadline '{ev.deadline_text}' not found in {ev.segment_id} - discarded")

    if owner_source != "stated":
        for h in t.name_hints:
            seg = segmap.get(h.segment_id)
            name = find_verbatim(h.name, seg.text, thr) if seg else None
            if name:
                hint = f"Possibly {name} - inferred from “{h.quote}” ({h.segment_id}); not stated as the owner."
                annotation = f"{annotation} {hint}" if annotation else hint
                break

    seg_ids = [t.segment_id] + [e.segment_id for e in t.history] + [e.segment_id for e in t.owner_evidence] + [
        e.segment_id for e in t.deadline_evidence
    ]
    quotes = [t.quote] + [e.quote for e in t.history]
    return ActionItem(
        id=f"A{idx}", task=t.description, owner=owner, owner_source=owner_source, owner_annotation=annotation,
        deadline=deadline, kind=t.kind, status=t.status,
        provenance=_provenance(seg_ids, quotes, segmap, order), verification_notes=notes + t.notes,
    )


class NLIChecker:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()  # CPU is fine
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.entail = next(i for i, l in labels.items() if l.startswith("entail"))

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        with self.torch.inference_mode():
            enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            probs = self.model(**enc).logits.softmax(-1)[0]
        return float(probs[self.entail])


def verify(
    life: LifecycleResult,
    segments: list[RefinedSegment],
    diarized: bool,
    settings: Settings,
    progress: Progress = lambda *a: None,
) -> VerifiedOutputs:
    progress(STAGE, "Checking owners, deadlines and provenance", None)
    segmap = {s.id: s for s in segments}
    order = {s.id: i for i, s in enumerate(segments)}
    thr = settings.quote_match_threshold
    warnings: list[str] = []

    decisions: list[Decision] = []
    rejected, deferred, unresolved = [], [], []
    for p in life.proposals:
        if p.status == "accepted":
            acc = next(e for e in reversed(p.history) if e.status == "accepted")
            decisions.append(
                Decision(
                    id=f"D{len(decisions) + 1}", decision=p.description, proposed_by=p.proposed_by,
                    accepted_by=acc.speaker if diarized else None, source_proposal=p.proposal_id,
                    provenance=_provenance([e.segment_id for e in p.history], [e.quote for e in p.history], segmap, order),
                    history=p.history,
                )
            )
        else:
            {"rejected": rejected, "deferred": deferred}.get(p.status, unresolved).append(_proposal_summary(p, segmap, order))

    actions, unconfirmed = [], []
    for t in life.tasks:
        if t.status == "declined":
            continue
        target = actions if t.status == "confirmed" else unconfirmed
        item = _verify_task(t, len(actions) + len(unconfirmed) + 1, segmap, order, diarized, thr)
        target.append(item)
    for i, a in enumerate(actions, 1):
        a.id = f"A{i}"
    for i, a in enumerate(unconfirmed, 1):
        a.id = f"R{i}"

    nli_model = None
    if settings.nli_check and (decisions or actions):
        try:
            progress(STAGE, f"NLI support check ({settings.nli_model})", None)
            nli = NLIChecker(settings.nli_model)
            for item, claim in [(d, d.decision) for d in decisions] + [(a, a.task) for a in actions]:
                premise = " ".join(segmap[s].text for s in item.provenance.segment_ids)
                p = nli.entail_prob(premise, claim)
                if p < settings.nli_threshold:
                    item.flags.append(f"weak NLI support (entailment p={p:.2f}) - please review")
            nli_model = settings.nli_model
        except Exception as e:
            warnings.append(f"NLI check skipped ({type(e).__name__}: {e})")

    return VerifiedOutputs(decisions, actions, unconfirmed, rejected, deferred, unresolved, nli_model, warnings)
