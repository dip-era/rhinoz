"""Stage 0 - input validation and audio decoding.

Uses PyAV (bundled with faster-whisper), so no system ffmpeg install is needed.
Every failure becomes a PipelineError with a message the UI shows verbatim.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from .errors import PipelineError

SAMPLE_RATE = 16000
SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".aac", ".wma"}
MIN_SECONDS = 0.5
SILENCE_RMS = 1e-4


def load_audio(path: str | Path) -> tuple[np.ndarray, dict]:
    """Validate a file and return (mono float32 16 kHz waveform, metadata)."""
    p = Path(path)
    if not p.exists():
        raise PipelineError(f"File not found: {p.name}", stage="input")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise PipelineError(
            f"Unsupported file format '{ext or '(none)'}'. Please upload one of: "
            + ", ".join(sorted(SUPPORTED_EXT)),
            stage="input",
        )
    if p.stat().st_size == 0:
        raise PipelineError("The uploaded file is empty (0 bytes).", stage="input")

    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise PipelineError("PyAV is not installed (it comes with faster-whisper: pip install faster-whisper).", "input") from e

    meta: dict = {"file": p.name, "format": ext.lstrip(".")}
    try:
        with av.open(str(p)) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise PipelineError("The file contains no audio track.", stage="input")
            st = streams[0]
            meta.update(
                codec=st.codec_context.name,
                source_sample_rate=st.codec_context.sample_rate,
                channels=getattr(st.codec_context, "channels", None),
            )
            if container.duration:
                meta["container_duration_sec"] = container.duration / 1_000_000
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(
            f"The file could not be read as audio - it may be corrupted or mislabelled ({type(e).__name__}).",
            stage="input",
        ) from e

    try:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(p), sampling_rate=SAMPLE_RATE)
    except Exception as e:
        raise PipelineError(f"The audio stream could not be decoded ({type(e).__name__}: {e}).", stage="input") from e

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < SAMPLE_RATE * MIN_SECONDS:
        raise PipelineError(
            f"The recording is too short or empty ({audio.size / SAMPLE_RATE:.2f}s of audio decoded).", stage="input"
        )
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < SILENCE_RMS:
        raise PipelineError("The recording appears to be completely silent.", stage="input")
    meta["duration_sec"] = audio.size / SAMPLE_RATE
    meta["rms"] = rms
    return audio, meta


def _to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_to_int16(audio).tobytes())


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        ch = w.getnchannels()
    a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def clip_wav_bytes(audio: np.ndarray, start: float, end: float, pad: float = 0.75, sr: int = SAMPLE_RATE) -> bytes:
    """In-memory WAV clip for the UI's 'play the evidence' buttons."""
    s = max(0, int((start - pad) * sr))
    e = min(len(audio), int((end + pad) * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_to_int16(audio[s:e]).tobytes())
    return buf.getvalue()
