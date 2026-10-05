"""Offline threshold tuning for the acoustic verifier - no models or LLM calls needed.

Every edit's Δlogp, ASR confidence and glossary support are stored in record.json, so
verdicts can be recomputed for any (tau_conf, tau_glossary, strong_margin) grid.

    python -m eval.tune_thresholds outputs/<run1> outputs/<run2>

Keep the grid coarse: 2-3 recordings is a tiny dev set and easy to overfit.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from meeting_assistant.config import ROOT
from meeting_assistant.record import load_record
from meeting_assistant.refine import Thresholds, apply_edits, finalize

from . import metrics
from .run_eval import reference_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--keys", type=Path, default=ROOT / "eval" / "answer_keys")
    args = ap.parse_args()
    keys = {json.loads(p.read_text(encoding="utf-8"))["name"]: json.loads(p.read_text(encoding="utf-8"))
            for p in args.keys.glob("*.json")}

    data = []
    for d in args.runs:
        rec = load_record(d / "record.json")
        name = next((n for n in keys if n in d.name or n in rec.source_file), None)
        if name:
            data.append((rec, keys[name]))
    if not data:
        raise SystemExit("no runs matched an answer key")

    grid = itertools.product([0.0, 2.0, 4.0, 6.0, 8.0, 12.0], [0.0, 1.5, 3.0, 5.0], [0.0, 0.5, 1.0])
    results = []
    for tau_conf, tau_gl, strong in grid:
        tot_terr, tot_viol, tot_wer = 0.0, 0, 0.0
        for rec, key in data:
            s = rec.settings_snapshot
            thr = Thresholds(s.get("low_conf", 0.5), s.get("phonetic_threshold", 0.8), tau_conf, tau_gl, strong,
                             s.get("no_acoustic_min_llm_conf", 0.75))
            verdicts = [v.model_copy() for v in rec.edits]
            finalize(verdicts, thr, rec.models.acoustic_verifier is not None)
            refined = " ".join(x.text for x in apply_edits(rec.raw_transcript, verdicts))
            raw = " ".join(x.text for x in rec.raw_transcript.segments)
            ref = reference_text(key)
            tot_terr += metrics.term_error(ref, refined, key.get("glossary_terms_to_score", []))["term_error_rate"]
            names = [n.strip() for n in key.get("attendees", "").split(",") if n.strip()]
            tot_viol += metrics.protected_violations(ref, raw, refined, names)["violations"]
            tot_wer += metrics.wer(ref, refined)
        n = len(data)
        results.append((tot_viol, tot_terr / n, tot_wer / n, tau_conf, tau_gl, strong))

    results.sort()
    print("violations | mean term err | mean WER | tau_conf | tau_glossary | strong_margin")
    for r in results[:15]:
        print(f"{r[0]:10d} | {r[1]:13.1%} | {r[2]:8.1%} | {r[3]:8.1f} | {r[4]:12.1f} | {r[5]:13.1f}")
    print("\nPick the simplest setting among the top rows; put it in .env (TAU_CONF, TAU_GLOSSARY, STRONG_MARGIN).")


if __name__ == "__main__":
    main()
