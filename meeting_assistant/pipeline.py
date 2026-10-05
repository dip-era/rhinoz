"""End-to-end orchestration: Stage 0 -> 1 -> 2 -> 3a -> 4 -> 3b -> record.

Models are loaded one at a time and freed before the next (6 GB VRAM budget).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import asr, audio_io, documentation, record as record_mod, refine, verification
from .config import Settings
from .errors import PipelineError
from .glossary import parse_term_list
from .llm_client import LLMClient
from .schemas import MeetingRecord

Progress = Callable[[str, str, float | None], None]


@dataclass
class PipelineResult:
    record: MeetingRecord
    output_dir: Path
    audio_wav: Path
    files: dict[str, Path] = field(default_factory=dict)


def run_pipeline(
    input_path: str | Path,
    glossary_text: str = "",
    attendees_text: str = "",
    settings: Settings | None = None,
    progress: Progress | None = None,
) -> PipelineResult:
    settings = settings or Settings.from_env()
    progress = progress or (lambda stage, msg, frac=None: print(f"[{stage}] {msg}"))
    warnings: list[str] = []
    input_path = Path(input_path)

    # ---- Stage 0: input ----------------------------------------------------
    progress("Stage 0 · Input", f"Validating {input_path.name}", None)
    audio, meta = audio_io.load_audio(input_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", input_path.stem)[:40] or "meeting"
    out_dir = settings.output_dir / f"{stamp}_{safe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio_16k.wav"
    audio_io.write_wav(wav_path, audio)
    progress("Stage 0 · Input", f"OK: {meta['duration_sec']:.1f}s of audio ({meta.get('codec')})", 1.0)

    # Create LLM clients before the slow ASR so a missing key fails fast.
    cache = settings.cache_dir if settings.use_llm_cache else None
    llm1 = LLMClient(settings.llm1_model, settings.groq_api_key, "LLM #1 (refinement)", cache, settings.llm1_tpm)
    llm2 = LLMClient(
        settings.llm2_model, settings.groq_api_key, "LLM #2 (documentation)", cache, settings.llm2_tpm,
        reasoning_effort=settings.llm2_reasoning_effort,
    )
    if settings.llm1_model == settings.llm2_model:
        warnings.append("LLM #1 and LLM #2 are the same model; the design expects two different models.")

    # ---- Stage 1: ASR (+ optional diarization) -----------------------------
    user_terms = parse_term_list(glossary_text)
    attendees = parse_term_list(attendees_text)
    hotwords = ", ".join(user_terms + attendees) or None
    transcript = asr.transcribe(audio, settings, hotwords, progress)
    if not transcript.segments:
        raise PipelineError("No speech was detected in the recording.", stage="asr")
    if settings.diarize:
        try:
            from . import diarization

            turns = diarization.diarize(audio, settings, progress)
            transcript = diarization.assign_speakers(transcript, turns)
            progress("Stage 1 · Diarization", f"{len({t[2] for t in turns})} speakers found", 1.0)
        except Exception as e:
            warnings.append(f"Diarization skipped ({type(e).__name__}: {e}); self-commitment owners will be 'unspecified'.")
            progress("Stage 1 · Diarization", f"skipped: {e}", None)

    # ---- Stage 2: refinement (LLM #1 + acoustic verification) ---------------
    ref = refine.refine(transcript, audio, user_terms, attendees, settings, llm1, progress)
    warnings += ref.warnings

    # ---- Stage 3a: speech acts + lifecycle (LLM #2) ------------------------
    life = documentation.extract_lifecycle(ref.refined_segments, transcript.diarized, llm2, settings, progress)
    warnings += life.warnings

    # ---- Stage 4: verification ---------------------------------------------
    ver = verification.verify(life, ref.refined_segments, transcript.diarized, settings, progress)
    warnings += ver.warnings

    # ---- Stage 3b: summary + minutes, consistent with verified lists -------
    summary, minutes, w = documentation.summarize(
        ref.refined_segments, transcript.diarized, ver.decisions, ver.action_items,
        ver.rejected + ver.deferred + ver.unresolved, llm2, settings, progress,
    )
    warnings += w

    rec = record_mod.build_record(
        source_file=input_path.name, transcript=transcript, refinement=ref, lifecycle=life, verified=ver,
        summary=summary, minutes=minutes, settings=settings, warnings=warnings,
    )
    files = record_mod.save_outputs(rec, out_dir)
    progress("Done", f"Outputs written to {out_dir}", 1.0)
    return PipelineResult(record=rec, output_dir=out_dir, audio_wav=wav_path, files=files)
