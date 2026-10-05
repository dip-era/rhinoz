"""(NICE) Raw ASR WER on a sample of AMI utterances (edinburghcstr/ami, IHM test split).

    pip install datasets soundfile
    python -m eval.ami_wer --n 200

Measures Stage 1 only. AMI decision annotations are deliberately NOT used (too costly to parse/match).
"""
from __future__ import annotations

import argparse

import numpy as np

from meeting_assistant.config import Settings

from .metrics import normalize_tokens, wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--config", default="ihm")
    args = ap.parse_args()

    from datasets import load_dataset
    from faster_whisper import WhisperModel

    try:
        ds = load_dataset("edinburghcstr/ami", args.config, split="test", streaming=True)
    except Exception as e:
        raise SystemExit(f"Could not load AMI ({e}). Newer `datasets` versions may need `pip install 'datasets<4'`.")
    s = Settings.from_env()
    try:
        import torch  # noqa: F401  (loads CUDA DLLs for ctranslate2 on Windows)
    except ImportError:
        pass
    model = WhisperModel(s.asr_model, device="auto", compute_type=s.asr_compute_type)
    refs, hyps = [], []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        a = ex["audio"]
        audio = np.asarray(a["array"], dtype=np.float32)
        if a["sampling_rate"] != 16000:
            n = int(len(audio) * 16000 / a["sampling_rate"])
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        segs, _ = model.transcribe(audio, language="en", beam_size=s.asr_beam_size, vad_filter=True)
        refs.append(ex["text"])
        hyps.append(" ".join(x.text for x in segs))
    ref_all, hyp_all = " ".join(refs), " ".join(hyps)
    print(f"AMI {args.config} test, {len(refs)} utterances, {len(normalize_tokens(ref_all))} ref words: "
          f"WER = {wer(ref_all, hyp_all):.1%}")


if __name__ == "__main__":
    main()
