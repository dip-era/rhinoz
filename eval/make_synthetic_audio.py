"""Render an answer key's script to a WAV with offline TTS (pyttsx3 / Windows SAPI voices),
or print the script for humans to read aloud.

    python -m eval.make_synthetic_audio eval/answer_keys/meeting1_platform_sync.json
    python -m eval.make_synthetic_audio eval/answer_keys/meeting1_platform_sync.json --print-script

TTS audio is only a SMOKE TEST (day-1 end-to-end check). Report results on real human
recordings - TTS has no accents, crosstalk or disfluencies, so it flatters the ASR.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from meeting_assistant.audio_io import SAMPLE_RATE, read_wav, write_wav
from meeting_assistant.config import ROOT


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return a
    n = int(len(a) * SAMPLE_RATE / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", type=Path)
    ap.add_argument("--print-script", action="store_true")
    ap.add_argument("--gap", type=float, default=0.5, help="silence between turns (s)")
    args = ap.parse_args()
    key = json.loads(args.key.read_text(encoding="utf-8"))

    if args.print_script:
        print(f"# {key['name']}\n\n{key.get('description', '')}\n")
        for t in key["turns"]:
            print(f"**{t['speaker']}:** {t['text']}\n")
        return

    import pyttsx3
    import wave

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    speakers = list(dict.fromkeys(t["speaker"] for t in key["turns"]))
    tmp = Path(tempfile.mkdtemp())
    files = []
    for i, t in enumerate(key["turns"]):
        engine.setProperty("voice", voices[speakers.index(t["speaker"]) % len(voices)].id)
        engine.setProperty("rate", 175)
        f = tmp / f"turn_{i:03d}.wav"
        engine.save_to_file(t["text"], str(f))
        files.append(f)
    engine.runAndWait()

    parts = []
    gap = np.zeros(int(args.gap * SAMPLE_RATE), dtype=np.float32)
    for f in files:
        with wave.open(str(f), "rb") as w:
            sr = w.getframerate()
        parts += [_resample(read_wav(f), sr), gap]
    out = ROOT / key["audio"]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, np.concatenate(parts))
    print(f"wrote {out} ({sum(len(p) for p in parts) / SAMPLE_RATE:.1f}s, {len(voices)} voices available)")


if __name__ == "__main__":
    main()
