"""Run the pipeline on every scripted recording that has an answer key, score it,
and write eval/results.md (the table that goes into the README).

    python -m eval.run_eval                     # run pipeline + score
    python -m eval.run_eval --reuse outputs/20261006_101500_meeting1_platform_sync  (score an existing run; repeatable)
    python -m eval.run_eval --no-acoustic       # ablation: acoustic verification off
    python -m eval.run_eval --no-user-glossary  # only LLM-inferred glossary (harder)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meeting_assistant.config import ROOT, Settings
from meeting_assistant.errors import PipelineError
from meeting_assistant.pipeline import run_pipeline
from meeting_assistant.record import load_record

from . import metrics


def reference_text(key: dict) -> str:
    return " ".join(t["text"] for t in key["turns"])


def score(record, key: dict) -> dict:
    ref = reference_text(key)
    raw = " ".join(s.text for s in record.raw_transcript.segments)
    refined = " ".join(s.text for s in record.refined_transcript)
    names = [n.strip() for n in key.get("attendees", "").split(",") if n.strip()]
    terms = key.get("glossary_terms_to_score", [])
    te_raw, te_ref = metrics.term_error(ref, raw, terms), metrics.term_error(ref, refined, terms)
    dec = metrics.decision_metrics([d.decision for d in record.decisions], key["decisions"], key.get("not_decisions", []))
    act = metrics.action_metrics([a.model_dump() for a in record.action_items], key["action_items"])
    return {
        "name": key["name"],
        "wer_raw": metrics.wer(ref, raw),
        "wer_refined": metrics.wer(ref, refined),
        "term_err_raw": te_raw["term_error_rate"],
        "term_err_refined": te_ref["term_error_rate"],
        "term_detail": {"raw": te_raw, "refined": te_ref},
        "protected": metrics.protected_violations(ref, raw, refined, names),
        "edits_accepted": sum(e.accepted for e in record.edits),
        "edits_proposed": len(record.edits),
        "decisions": dec,
        "actions": act,
        "acoustic_verifier": record.models.acoustic_verifier,
    }


def table(rows: list[dict]) -> str:
    h = ("| Recording | WER raw | WER refined | Term err raw | Term err refined | Protected violations | Edits acc/prop "
         "| Decision P | Decision R | Proposal leaks | Action P | Action R | Invented owners | Invented deadlines |")
    lines = [h, "|" + "---|" * 14]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['wer_raw']:.1%} | {r['wer_refined']:.1%} | {r['term_err_raw']:.1%} | "
            f"{r['term_err_refined']:.1%} | {r['protected']['violations']} | {r['edits_accepted']}/{r['edits_proposed']} | "
            f"{r['decisions']['precision']:.2f} | {r['decisions']['recall']:.2f} | {r['decisions']['proposal_leaks']} | "
            f"{r['actions']['precision']:.2f} | {r['actions']['recall']:.2f} | {r['actions']['invented_owners']} | "
            f"{r['actions']['invented_deadlines']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=Path, default=ROOT / "eval" / "answer_keys")
    ap.add_argument("--reuse", type=Path, action="append", default=[], help="existing output dir(s) to score")
    ap.add_argument("--no-acoustic", action="store_true")
    ap.add_argument("--no-user-glossary", action="store_true")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results.md")
    args = ap.parse_args()

    keys = {json.loads(p.read_text(encoding="utf-8"))["name"]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(args.keys.glob("*.json"))}
    s = Settings.from_env()
    s.acoustic_check = not args.no_acoustic
    s.diarize = s.diarize or args.diarize

    rows = []
    if args.reuse:
        for d in args.reuse:
            rec = load_record(d / "record.json")
            name = next((n for n in keys if n in d.name or n in rec.source_file), None)
            if not name:
                print(f"skip {d}: no matching answer key")
                continue
            rows.append(score(rec, keys[name]))
    else:
        for name, key in keys.items():
            audio = ROOT / key["audio"]
            if not audio.exists():
                print(f"skip {name}: recording not found at {audio}")
                continue
            print(f"=== {name} ===")
            try:
                res = run_pipeline(audio, "" if args.no_user_glossary else key.get("user_glossary", ""),
                                   key.get("attendees", ""), s)
            except PipelineError as e:
                print(f"  FAILED: {e}")
                continue
            rows.append(score(res.record, key))
            print(f"  outputs: {res.output_dir}")

    if not rows:
        print("Nothing scored.")
        return
    detail_dir = args.out.parent / "results"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (detail_dir / f"{r['name']}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    cfg = f"acoustic={'off' if args.no_acoustic else 'on'}, user_glossary={'off' if args.no_user_glossary else 'on'}"
    md = f"# Evaluation results ({cfg})\n\n{table(rows)}\n"
    args.out.write_text(md, encoding="utf-8")
    print("\n" + md)
    for r in rows:
        print(f"[{r['name']}] decision matches: {r['decisions']['matches']}")
        print(f"[{r['name']}] action matches:   {r['actions']['matches']}")
        for d in r["actions"]["details"]:
            print(f"[{r['name']}]   ! {d}")
        if r["protected"]["broken_tokens"]:
            print(f"[{r['name']}]   ! protected tokens broken by refinement: {r['protected']['broken_tokens']}")


if __name__ == "__main__":
    main()
