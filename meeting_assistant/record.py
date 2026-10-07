"""Canonical JSON record + renderers. The Markdown is generated FROM the JSON record,
so the human-readable and machine-readable outputs always agree."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .documentation import LifecycleResult
from .refine import RefinementResult
from .schemas import SpeakerIdentity, MeetingRecord, MinutesSection, ModelInfo, RefinedSegment, Transcript
from .utils import fmt_ts
from .verification import VerifiedOutputs


def build_record(
    *,
    source_file: str,
    transcript: Transcript,
    refinement: RefinementResult,
    lifecycle: LifecycleResult,
    verified: VerifiedOutputs,
    summary: str,
    minutes: list[MinutesSection],
    settings: Settings,
    warnings: list[str],
    speakers: list[SpeakerIdentity] | None = None,
) -> MeetingRecord:
    snap = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(settings).items()}
    for secret in ("groq_api_key", "hf_token"):
        snap.pop(secret, None)
    return MeetingRecord(
        source_file=source_file,
        created_at=datetime.now().isoformat(timespec="seconds"),
        duration_sec=round(transcript.duration, 2),
        diarized=transcript.diarized,
        models=ModelInfo(
            asr=transcript.asr_model,
            diarization=settings.diarization_model if transcript.diarized else None,
            acoustic_verifier=refinement.acoustic_model if refinement.acoustic_used else None,
            refiner_llm=f"groq/{settings.llm1_model}",
            supervisor_llm=f"groq/{refinement.supervisor_model}" if refinement.supervisor_model else None,
            documenter_llm=f"groq/{settings.llm2_model}",
            nli=verified.nli_model,
        ),
        summary=summary,
        minutes=minutes,
        decisions=verified.decisions,
        action_items=verified.action_items,
        unconfirmed_requests=verified.unconfirmed_requests,
        rejected_proposals=verified.rejected,
        deferred_proposals=verified.deferred,
        unresolved_proposals=verified.unresolved,
        speakers=speakers or [],
        raw_transcript=transcript,
        refined_transcript=refinement.refined_segments,
        glossary=refinement.glossary,
        candidates=refinement.candidates,
        edits=refinement.verdicts,
        speech_acts=lifecycle.speech_acts,
        settings_snapshot=snap,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
def speaker_turns(segments, diarized: bool, max_gap: float = 2.0) -> list[list]:
    """Group consecutive segments into paragraphs: one per speaker turn when diarized, else split at pauses."""
    turns: list[list] = []
    for s in segments:
        prev = turns[-1][-1] if turns else None
        same = prev is not None and (s.speaker == prev.speaker if diarized else s.start - prev.end <= max_gap)
        if same:
            turns[-1].append(s)
        else:
            turns.append([s])
    return turns


def transcript_text(segments, diarized: bool, names: dict[str, str] | None = None) -> str:
    """One paragraph per speaker turn; speakers shown by resolved name (and role) when known, else SPEAKER_xx.
    The segment-id range keeps every paragraph traceable to the cited segments in the record."""
    paras = []
    for turn in speaker_turns(segments, diarized):
        first, last = turn[0], turn[-1]
        ids = first.id if first is last else f"{first.id}-{last.id}"
        spk = f" {(names or {}).get(first.speaker, first.speaker)}:" if diarized and first.speaker else ""
        text = " ".join(s.text.strip() for s in turn)
        paras.append(f"[{fmt_ts(first.start)} - {fmt_ts(last.end)}] ({ids}){spk} {text}")
    return "\n\n".join(paras) + "\n"


def _cite(prov) -> str:
    if not prov.segment_ids:
        return ""
    return f"{', '.join(prov.segment_ids)} @ {fmt_ts(prov.start)}–{fmt_ts(prov.end)}"


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _names(rec: MeetingRecord) -> dict[str, str]:
    return {s.label: s.display_name for s in rec.speakers}


def to_markdown(rec: MeetingRecord) -> str:
    names = _names(rec)
    L: list[str] = []
    L.append(f"# Meeting record - {rec.source_file}")
    L.append("")
    L.append(f"_Generated {rec.created_at} · duration {fmt_ts(rec.duration_sec)} · "
             f"speakers {'diarized' if rec.diarized else 'not identified'}_")
    L.append("")
    if rec.speakers:
        L.append("## Speakers")
        L.append("| Label | Name | Confidence | Evidence |")
        L.append("|---|---|---|---|")
        for sp in rec.speakers:
            ev = "; ".join(f"{e.kind} {e.segment_id}: “{e.quote}”" for e in sp.evidence[:3]) or "-"
            shown = sp.display_name if sp.name else "not mentioned"
            L.append(f"| {sp.label} | {shown} | {sp.confidence} | {_md_escape(ev)} |")
        L.append("")
    L.append("## Summary")
    L.append(rec.summary or "_No summary produced._")
    L.append("")
    L.append("## Minutes")
    if not rec.minutes:
        L.append("_No minutes produced._")
    for sec in rec.minutes:
        L.append(f"### {sec.topic}")
        for p in sec.points:
            cite = f" _({', '.join(p.segment_ids)})_" if p.segment_ids else " _(uncited)_"
            L.append(f"- {p.text}{cite}")
        L.append("")
    L.append("## Key decisions")
    if not rec.decisions:
        L.append("_No decisions were reached._")
    else:
        L.append("| ID | Decision | How | By | Evidence |")
        L.append("|---|---|---|---|---|")
        for d in rec.decisions:
            flags = f" ⚠ {'; '.join(d.flags)}" if d.flags else ""
            how = "announced" if d.basis == "announced" else "agreed"
            by = names.get(d.proposed_by, d.proposed_by) if d.proposed_by else "unspecified"
            L.append(f"| {d.id} | {_md_escape(d.decision)}{flags} | {how} | {by} | {_cite(d.provenance)} |")
        for d in rec.decisions:
            for q in d.provenance.quotes:
                L.append(f"> {d.id}: “{q}”")
    L.append("")
    L.append("## Action items")
    if not rec.action_items:
        L.append("_No action items were assigned._")
    else:
        L.append("| ID | Task | Owner | Deadline | Evidence |")
        L.append("|---|---|---|---|---|")
        for a in rec.action_items:
            owner = a.owner
            flags = f" ⚠ {'; '.join(a.flags)}" if a.flags else ""
            L.append(f"| {a.id} | {_md_escape(a.task)}{flags} | {owner} | {_md_escape(a.deadline)} | {_cite(a.provenance)} |")
        notes = [a for a in rec.action_items if a.owner_annotation]
        if notes:
            L.append("")
            L.append("_Owner annotations (context only - not stated owners):_")
            for a in notes:
                L.append(f"- {a.id}: {a.owner_annotation}")
    L.append("")
    if rec.unconfirmed_requests:
        L.append("## Requests not confirmed in the meeting")
        for r in rec.unconfirmed_requests:
            L.append(f"- {r.id}: {r.task} (owner: {r.owner}; deadline: {r.deadline}) - {_cite(r.provenance)}")
        L.append("")
    not_adopted = rec.rejected_proposals + rec.deferred_proposals + rec.unresolved_proposals
    if not_adopted:
        L.append("## Proposals not adopted")
        for p in not_adopted:
            L.append(f"- **{p.status}** - {p.proposal} ({_cite(p.provenance)})")
        L.append("")
    accepted = [e for e in rec.edits if e.accepted]
    L.append("## Transcript refinement")
    L.append(f"{len(accepted)} of {len(rec.edits)} proposed edits accepted.")
    for e in accepted:
        L.append(f"- {e.segment_id}: “{e.located_text}” → “{e.replacement}” ({e.edit_type}; {e.verdict_reason})")
    L.append("")
    L.append("## Models")
    m = rec.models
    L.append(f"- Speech-to-text: {m.asr}")
    if m.diarization:
        L.append(f"- Diarization: {m.diarization}")
    L.append(f"- LLM #1 (refinement): {m.refiner_llm}")
    L.append(f"- LLM #3 (refinement supervisor): {m.supervisor_llm or 'not used'}")
    L.append(f"- Acoustic verifier: {m.acoustic_verifier or 'not used'}")
    L.append(f"- LLM #2 (documentation): {m.documenter_llm}")
    if m.nli:
        L.append(f"- NLI flagger: {m.nli}")
    return "\n".join(L) + "\n"


def save_outputs(rec: MeetingRecord, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "record.json": out_dir / "record.json",
        "record.md": out_dir / "record.md",
        "raw_transcript.txt": out_dir / "raw_transcript.txt",
        "refined_transcript.txt": out_dir / "refined_transcript.txt",
    }
    files["record.json"].write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    files["record.md"].write_text(to_markdown(rec), encoding="utf-8")
    files["raw_transcript.txt"].write_text(transcript_text(rec.raw_transcript.segments, rec.diarized, _names(rec)), encoding="utf-8")
    files["refined_transcript.txt"].write_text(transcript_text(rec.refined_transcript, rec.diarized, _names(rec)), encoding="utf-8")
    return files


def load_record(path: str | Path) -> MeetingRecord:
    return MeetingRecord.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
