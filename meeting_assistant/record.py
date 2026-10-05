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
from .schemas import MeetingRecord, MinutesSection, ModelInfo, RefinedSegment, Transcript
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
def transcript_text(segments, diarized: bool) -> str:
    lines = []
    for s in segments:
        spk = f" {s.speaker}:" if diarized and s.speaker else ""
        lines.append(f"[{fmt_ts(s.start)} - {fmt_ts(s.end)}] ({s.id}){spk} {s.text}")
    return "\n".join(lines) + "\n"


def _cite(prov) -> str:
    if not prov.segment_ids:
        return ""
    return f"{', '.join(prov.segment_ids)} @ {fmt_ts(prov.start)}–{fmt_ts(prov.end)}"


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def to_markdown(rec: MeetingRecord) -> str:
    L: list[str] = []
    L.append(f"# Meeting record - {rec.source_file}")
    L.append("")
    L.append(f"_Generated {rec.created_at} · duration {fmt_ts(rec.duration_sec)} · "
             f"speakers {'diarized' if rec.diarized else 'not identified'}_")
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
        L.append("| ID | Decision | Proposed by | Evidence |")
        L.append("|---|---|---|---|")
        for d in rec.decisions:
            flags = f" ⚠ {'; '.join(d.flags)}" if d.flags else ""
            L.append(f"| {d.id} | {_md_escape(d.decision)}{flags} | {d.proposed_by or 'unspecified'} | {_cite(d.provenance)} |")
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
            owner = a.owner + (" _(speaker label)_" if a.owner_source == "speaker_label" else "")
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
    files["raw_transcript.txt"].write_text(transcript_text(rec.raw_transcript.segments, rec.diarized), encoding="utf-8")
    files["refined_transcript.txt"].write_text(transcript_text(rec.refined_transcript, rec.diarized), encoding="utf-8")
    return files


def load_record(path: str | Path) -> MeetingRecord:
    return MeetingRecord.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
