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
    # pyannote 4: the "exclusive" version (one speaker at a time) is meant for aligning with transcript words
    ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(out, "speaker_diarization", out)
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


def _sentence_units(words) -> list[list[int]]:
    """Split the word stream into sentence-like units: at . ? ! , at pauses > 0.8 s, and at speaker changes."""
    units, cur = [], []
    for i, w in enumerate(words):
        if cur and (w.speaker != words[cur[-1]].speaker or w.start - words[cur[-1]].end > 0.8):
            units.append(cur)
            cur = []
        cur.append(i)
        if w.text.rstrip().endswith((".", "?", "!")):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


def recheck_speakers(words, audio: np.ndarray, turns: list[tuple[float, float, str]], settings: Settings,
                     progress: Progress = lambda *a: None) -> int:
    """Voice check per sentence. pyannote can fold a short interjection (e.g. a one-second question) into the
    long turn around it. Each speaker gets an average voice embedding from their own long turns; a sentence moves
    to another speaker only if it matches that speaker clearly better. Returns how many sentences moved."""
    import torch
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment as PSegment

    progress(STAGE, "Voice re-check of each sentence", None)
    try:
        model = Model.from_pretrained(settings.embedding_model, token=settings.hf_token)
    except TypeError:
        model = Model.from_pretrained(settings.embedding_model, use_auth_token=settings.hf_token)
    inf = Inference(model, window="whole")
    if torch.cuda.is_available():
        inf.to(torch.device("cuda"))
    wav = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    total = len(audio) / SAMPLE_RATE

    def emb(a: float, b: float) -> np.ndarray:
        v = np.asarray(inf.crop(wav, PSegment(max(0.0, a), min(total, b)))).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    centroids: dict[str, np.ndarray] = {}
    for spk in sorted({t[2] for t in turns}):
        chunks = [(x, min(x + 3.0, b)) for a, b, sp in turns if sp == spk and b - a >= 2.0
                  for x in np.arange(a, b - 1.5, 3.0)][:40]
        if len(chunks) >= 3:  # too little audio for a reliable voice profile otherwise
            c = np.mean([emb(a, b) for a, b in chunks], axis=0)
            centroids[spk] = c / (np.linalg.norm(c) + 1e-9)

    moved = 0
    units = _sentence_units(words)
    for ui, u in enumerate(units):
        a, b = words[u[0]].start, words[u[-1]].end
        if b - a < settings.recheck_min_dur or len(centroids) < 2:
            continue
        e = emb(a - 0.05, b + 0.05)
        sims = {spk: float(e @ c) for spk, c in centroids.items()}
        best = max(sims, key=sims.get)
        cur = words[u[0]].speaker
        if best != cur and sims[best] >= settings.recheck_min_sim and sims[best] - sims.get(cur, -1.0) >= settings.recheck_margin:
            for i in u:
                words[i].speaker = best
            moved += 1
        if ui % 50 == 0:
            progress(STAGE, f"Voice re-check {ui}/{len(units)} sentences", ui / max(len(units), 1))
    del inf, model
    free_gpu()
    return moved


def assign_speakers(transcript: Transcript, turns: list[tuple[float, float, str]], audio: np.ndarray | None = None,
                    settings: Settings | None = None, progress: Progress = lambda *a: None) -> Transcript:
    """Label each word, smooth single-word flips, voice-check each sentence, and split segments at speaker changes."""
    if not turns:
        return transcript
    words = [w.model_copy() for w in transcript.words]
    for w in words:
        w.speaker = _speaker_for(w.start, w.end, turns)
    for i in range(1, len(words) - 1):
        a, b, c = words[i - 1].speaker, words[i].speaker, words[i + 1].speaker
        if a == c and b != a:
            words[i].speaker = a
    if audio is not None and settings is not None and settings.diarization_recheck:
        try:
            moved = recheck_speakers(words, audio, turns, settings, progress)
            progress(STAGE, f"Voice re-check moved {moved} sentence(s) to a different speaker", 1.0)
        except Exception as e:  # the re-check is an improvement, never a reason to fail the run
            progress(STAGE, f"Voice re-check skipped ({type(e).__name__}: {e})", None)

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
