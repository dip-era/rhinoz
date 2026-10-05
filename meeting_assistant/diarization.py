"""Optional speaker diarization with pyannote, then word-level speaker assignment.

Calls pyannote directly (no WhisperX) to avoid WhisperX's pinned torch/ctranslate2
versions. Needs HF_TOKEN and accepting the model terms on huggingface.co.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .audio_io import SAMPLE_RATE
from .config import Settings
from .schemas import Segment, Transcript
from .utils import free_gpu

Progress = Callable[[str, str, float | None], None]
STAGE = "Stage 1 · Diarization"


def diarize(audio: np.ndarray, settings: Settings, progress: Progress = lambda *a: None) -> list[tuple[float, float, str]]:
    import torch
    from pyannote.audio import Pipeline

    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set (needed for pyannote's gated models)")
    progress(STAGE, f"Loading {settings.diarization_model}", None)
    try:
        pipe = Pipeline.from_pretrained(settings.diarization_model, token=settings.hf_token)  # pyannote >= 4
    except TypeError:
        pipe = Pipeline.from_pretrained(settings.diarization_model, use_auth_token=settings.hf_token)  # pyannote 3.x
    if pipe is None:
        raise RuntimeError("pyannote pipeline could not be loaded - did you accept the model terms on Hugging Face?")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    progress(STAGE, "Running speaker diarization", None)
    kwargs = {"num_speakers": settings.num_speakers} if settings.num_speakers else {}
    out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}, **kwargs)
    ann = getattr(out, "speaker_diarization", out)  # pyannote 4 returns a wrapper object
    turns = [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]
    del pipe
    free_gpu()
    return turns


def _speaker_for(start: float, end: float, turns: list[tuple[float, float, str]]) -> str | None:
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = min(end, te) - max(start, ts)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is not None:
        return best
    mid = (start + end) / 2  # no overlap: nearest turn
    nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])), default=None)
    return nearest[2] if nearest else None


def assign_speakers(transcript: Transcript, turns: list[tuple[float, float, str]]) -> Transcript:
    """Label each word, smooth single-word flips, and split segments at speaker changes."""
    if not turns:
        return transcript
    words = [w.model_copy() for w in transcript.words]
    for w in words:
        w.speaker = _speaker_for(w.start, w.end, turns)
    for i in range(1, len(words) - 1):
        a, b, c = words[i - 1].speaker, words[i].speaker, words[i + 1].speaker
        if a == c and b != a:
            words[i].speaker = a

    new_segments: list[Segment] = []
    for seg in transcript.segments:
        run: list[int] = []
        for wid in seg.word_ids:
            if run and words[wid].speaker != words[run[-1]].speaker:
                new_segments.append(_make_segment(run, words, seg))
                run = []
            run.append(wid)
        if run:
            new_segments.append(_make_segment(run, words, seg))
    for i, s in enumerate(new_segments, 1):
        s.id = f"S{i:04d}"
    return Transcript(
        words=words,
        segments=new_segments,
        duration=transcript.duration,
        language=transcript.language,
        asr_model=transcript.asr_model,
        diarized=True,
    )


def _make_segment(ids: list[int], words, parent: Segment) -> Segment:
    return Segment(
        id="",
        start=words[ids[0]].start,
        end=words[ids[-1]].end,
        speaker=words[ids[0]].speaker,
        word_ids=list(ids),
        text=" ".join(words[i].text for i in ids),
        avg_logprob=parent.avg_logprob,
        no_speech_prob=parent.no_speech_prob,
    )
