"""Stage 1 - ASR with faster-whisper: VAD first, word-level timestamps and probabilities."""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from .audio_io import SAMPLE_RATE
from .config import Settings
from .errors import PipelineError
from .schemas import Segment, Transcript, Word
from .utils import fmt_ts, free_gpu

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 1 · ASR"


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _load_model(settings: Settings, progress: Progress):
    # Importing torch first loads its bundled cuBLAS/cuDNN DLLs, which ctranslate2 needs on Windows.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    from faster_whisper import WhisperModel

    device = settings.asr_device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    compute = settings.asr_compute_type if device == "cuda" else "int8"
    progress(STAGE, f"Loading faster-whisper '{settings.asr_model}' on {device} ({compute})", None)
    try:
        return WhisperModel(settings.asr_model, device=device, compute_type=compute), device
    except Exception as e:
        if device == "cuda":
            progress(STAGE, f"CUDA load failed ({e}); falling back to CPU - this will be slow", None)
            try:
                return WhisperModel(settings.asr_model, device="cpu", compute_type="int8"), "cpu"
            except Exception as e2:
                raise PipelineError(f"Could not load the speech-to-text model: {e2}", stage="asr") from e2
        raise PipelineError(f"Could not load the speech-to-text model: {e}", stage="asr") from e


def transcribe(
    audio: np.ndarray,
    settings: Settings,
    hotwords: str | None = None,
    progress: Progress = lambda *a: None,
) -> Transcript:
    model, device = _load_model(settings, progress)
    duration = len(audio) / SAMPLE_RATE
    kwargs = dict(
        language="en",
        task="transcribe",
        beam_size=settings.asr_beam_size,
        vad_filter=True,  # VAD before ASR: no hallucinated text over silence
        word_timestamps=True,
        condition_on_previous_text=False,  # stops one hallucination propagating into the next window
    )
    if hotwords:
        params = inspect.signature(model.transcribe).parameters
        # `hotwords` is re-applied to every window; `initial_prompt` only to the first one.
        kwargs["hotwords" if "hotwords" in params else "initial_prompt"] = hotwords

    words: list[Word] = []
    segments: list[Segment] = []
    seg_iter = None
    try:
        seg_iter, _info = model.transcribe(audio, **kwargs)
        for fs in seg_iter:
            progress(STAGE, f"Transcribed {fmt_ts(fs.end)} / {fmt_ts(duration)}", min(fs.end / max(duration, 1e-6), 1.0))
            ids: list[int] = []
            for w in fs.words or []:
                text = w.word.strip()
                if not text:
                    continue
                ids.append(len(words))
                words.append(Word(id=len(words), text=text, start=float(w.start), end=float(w.end), prob=float(w.probability)))
            if not ids:
                continue
            segments.append(
                Segment(
                    id=f"S{len(segments) + 1:04d}",
                    start=words[ids[0]].start,
                    end=words[ids[-1]].end,
                    word_ids=ids,
                    text=" ".join(words[i].text for i in ids),
                    avg_logprob=float(fs.avg_logprob),
                    no_speech_prob=float(fs.no_speech_prob),
                )
            )
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Speech recognition failed on {device}: {type(e).__name__}: {e}", stage="asr") from e
    finally:
        del seg_iter, model
        free_gpu()

    return Transcript(words=words, segments=segments, duration=duration, asr_model=f"faster-whisper/{settings.asr_model}")
