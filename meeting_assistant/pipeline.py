"""Orchestration, split into stages the UI can run one at a time:

    stage0_validate   -> PipelineSession (audio decoded, output folder created)
    stage1_transcribe -> raw transcript            (faster-whisper [+ diarization])
    stage2_refine     -> refined transcript        (LLM #1 + acoustic verification)
    stage3_document   -> MeetingRecord             (LLM #2 lifecycle -> verification -> summary/minutes)

`run_pipeline` chains all four for the CLI and the evaluation scripts.
Models are loaded one at a time and freed before the next (6 GB VRAM budget).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import asr, audio_io, speakers as speakers_mod, documentation, record as record_mod, refine, verification
from .config import Settings
from .errors import PipelineError
from .glossary import parse_term_list
from .llm_client import LLMClient
from .record import transcript_text
from .refine import RefinementResult
from .schemas import MeetingRecord, SpeakerIdentity, Transcript

Progress = Callable[[str, str, float | None], None]


def _default_progress(stage: str, msg: str, frac: float | None = None) -> None:
    print(f"[{stage}] {msg}")


@dataclass
class PipelineSession:
    input_path: Path
    settings: Settings
    audio: np.ndarray
    meta: dict
    output_dir: Path
    audio_wav: Path
    user_terms: list[str] = field(default_factory=list)
    attendees: list[str] = field(default_factory=list)
    transcript: Transcript | None = None
    speakers: list[SpeakerIdentity] = field(default_factory=list)
    refinement: RefinementResult | None = None
    record: MeetingRecord | None = None
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def speaker_names(self) -> dict[str, str]:
        return speakers_mod.name_map(self.speakers)


@dataclass
class PipelineResult:
    record: MeetingRecord
    output_dir: Path
    audio_wav: Path
    files: dict[str, Path] = field(default_factory=dict)


def _llm1(s: Settings) -> LLMClient:
    cache = s.cache_dir if s.use_llm_cache else None
    return LLMClient(s.llm1_model, s.groq_api_key, "LLM #1 (refinement)", cache, s.llm1_tpm)


def _speaker_llm(s: Settings) -> LLMClient:
    cache = s.cache_dir if s.use_llm_cache else None
    return LLMClient(s.speaker_id_model or s.llm2_model, s.groq_api_key, "Speaker naming", cache, s.llm2_tpm,
                     reasoning_effort=s.llm2_reasoning_effort)


def _llm2(s: Settings) -> LLMClient:
    cache = s.cache_dir if s.use_llm_cache else None
    return LLMClient(s.llm2_model, s.groq_api_key, "LLM #2 (documentation)", cache, s.llm2_tpm,
                     reasoning_effort=s.llm2_reasoning_effort)


# ---------------------------------------------------------------------------
def stage0_validate(input_path: str | Path, settings: Settings | None = None,
                    progress: Progress | None = None) -> PipelineSession:
    """Validate + decode the upload. Raises PipelineError with a user-facing message."""
    settings = settings or Settings.from_env()
    progress = progress or _default_progress
    input_path = Path(input_path)
    progress("Stage 0 · Input", f"Validating {input_path.name}", None)
    audio, meta = audio_io.load_audio(input_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", input_path.stem)[:40] or "meeting"
    out_dir = settings.output_dir / f"{stamp}_{safe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio_16k.wav"
    audio_io.write_wav(wav_path, audio)
    progress("Stage 0 · Input", f"OK: {meta['duration_sec']:.1f}s of audio ({meta.get('codec')})", 1.0)
    return PipelineSession(input_path=input_path, settings=settings, audio=audio, meta=meta,
                           output_dir=out_dir, audio_wav=wav_path)


def stage1_transcribe(sess: PipelineSession, glossary_text: str = "", attendees_text: str = "",
                      progress: Progress | None = None) -> Transcript:
    progress = progress or _default_progress
    s = sess.settings
    sess.transcript = sess.refinement = sess.record = None  # later stages are now stale
    sess.speakers = []
    sess.warnings = []
    sess.user_terms = parse_term_list(glossary_text)
    sess.attendees = parse_term_list(attendees_text)
    hotwords = ", ".join(sess.user_terms + sess.attendees) or None
    transcript = asr.transcribe(sess.audio, s, hotwords, progress)
    if not transcript.segments:
        raise PipelineError("No speech was detected in the recording.", stage="asr")
    if s.diarize:
        try:
            from . import diarization

            turns = diarization.diarize(sess.audio, s, progress)
            transcript = diarization.assign_speakers(transcript, turns)
            progress("Stage 1 · Diarization", f"{len({t[2] for t in turns})} speakers found", 1.0)
        except Exception as e:
            sess.warnings.append(f"Diarization skipped ({type(e).__name__}: {e}); self-commitment owners will be 'unspecified'.")
            progress("Stage 1 · Diarization", f"skipped: {e}", None)
    if transcript.diarized:  # name speakers only from what the meeting itself says
        llm = None
        if s.speaker_naming and s.groq_api_key:
            llm = _speaker_llm(s)
        elif s.speaker_naming:
            sess.warnings.append("Speaker naming skipped: GROQ_API_KEY is not set; speakers keep SPEAKER_xx labels.")
        try:
            sess.speakers, w = speakers_mod.resolve_speakers(transcript, llm, s, sess.attendees, progress)
            sess.warnings += w
        except PipelineError as e:
            sess.speakers, _ = speakers_mod.resolve_speakers(transcript, None, s)
            sess.warnings.append(f"Speaker naming skipped ({e}); speakers keep SPEAKER_xx labels.")
    sess.transcript = transcript
    p = sess.output_dir / "raw_transcript.txt"
    p.write_text(transcript_text(transcript.segments, transcript.diarized, sess.speaker_names), encoding="utf-8")
    sess.files["raw_transcript.txt"] = p
    progress("Stage 1 · ASR", f"Raw transcript ready: {len(transcript.segments)} segments, {len(transcript.words)} words", 1.0)
    return transcript


def stage2_refine(sess: PipelineSession, progress: Progress | None = None) -> RefinementResult:
    progress = progress or _default_progress
    if sess.transcript is None:
        raise PipelineError("Generate the raw transcript (Stage 1) first.", stage="refinement")
    sess.refinement = sess.record = None
    ref = refine.refine(sess.transcript, sess.audio, sess.user_terms, sess.attendees, sess.settings,
                        _llm1(sess.settings), progress)
    sess.refinement = ref
    p = sess.output_dir / "refined_transcript.txt"
    p.write_text(transcript_text(ref.refined_segments, sess.transcript.diarized, sess.speaker_names), encoding="utf-8")
    sess.files["refined_transcript.txt"] = p
    return ref


def stage3_document(sess: PipelineSession, progress: Progress | None = None) -> MeetingRecord:
    progress = progress or _default_progress
    if sess.refinement is None or sess.transcript is None:
        raise PipelineError("Generate the refined transcript (Stage 2) first.", stage="documentation")
    s, tr, ref = sess.settings, sess.transcript, sess.refinement
    warnings = list(sess.warnings) + list(ref.warnings)
    if s.llm1_model == s.llm2_model:
        warnings.append("LLM #1 and LLM #2 are the same model; the problem statement expects two distinct models.")
    llm2 = _llm2(s)
    names = sess.speaker_names

    life = documentation.extract_lifecycle(ref.refined_segments, tr.diarized, llm2, s, progress, names)  # 3a
    warnings += life.warnings
    ver = verification.verify(life, ref.refined_segments, tr.diarized, s, progress,
                              speakers_mod.identity_map(sess.speakers))  # 4
    warnings += ver.warnings
    summary, minutes, w = documentation.summarize(  # 3b, consistent with the verified lists
        ref.refined_segments, tr.diarized, ver.decisions, ver.action_items,
        ver.rejected + ver.deferred + ver.unresolved, llm2, s, progress, names,
    )
    warnings += w
    rec = record_mod.build_record(
        source_file=sess.input_path.name, transcript=tr, refinement=ref, lifecycle=life, verified=ver,
        summary=summary, minutes=minutes, settings=s, warnings=warnings, speakers=sess.speakers,
    )
    sess.record = rec
    sess.files.update(record_mod.save_outputs(rec, sess.output_dir))
    progress("Done", f"Outputs written to {sess.output_dir}", 1.0)
    return rec


# ---------------------------------------------------------------------------
def run_pipeline(
    input_path: str | Path,
    glossary_text: str = "",
    attendees_text: str = "",
    settings: Settings | None = None,
    progress: Progress | None = None,
) -> PipelineResult:
    """All stages in one go (CLI / evaluation)."""
    settings = settings or Settings.from_env()
    if not settings.groq_api_key:  # fail fast, before the slow ASR
        raise PipelineError("GROQ_API_KEY is not set. Put it in .env.", stage="LLM")
    sess = stage0_validate(input_path, settings, progress)
    stage1_transcribe(sess, glossary_text, attendees_text, progress)
    stage2_refine(sess, progress)
    rec = stage3_document(sess, progress)
    return PipelineResult(record=rec, output_dir=sess.output_dir, audio_wav=sess.audio_wav, files=sess.files)
