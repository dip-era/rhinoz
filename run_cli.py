"""Command-line entry point (same pipeline as the UI; handy for evaluation and debugging).

    python run_cli.py meeting.wav --glossary "Kubernetes, Grafana" --attendees "Priya, Rahul" --diarize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meeting_assistant.config import Settings
from meeting_assistant.errors import PipelineError
from meeting_assistant.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Evidence-traced meeting assistant (CLI)")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--glossary", default="", help="comma/newline separated terms, or @file.txt")
    ap.add_argument("--attendees", default="", help="comma separated names, or @file.txt")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--no-acoustic", action="store_true", help="disable teacher-forced acoustic verification")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="output root directory")
    args = ap.parse_args()

    def read_arg(v: str) -> str:
        return Path(v[1:]).read_text(encoding="utf-8") if v.startswith("@") else v

    s = Settings.from_env()
    if args.diarize:
        s.diarize = True
    if args.no_acoustic:
        s.acoustic_check = False
    if args.nli:
        s.nli_check = True
    if args.out:
        s.output_dir = args.out
    try:
        res = run_pipeline(args.audio, read_arg(args.glossary), read_arg(args.attendees), s)
    except PipelineError as e:
        print(f"ERROR [{e.stage}]: {e}", file=sys.stderr)
        return 2
    r = res.record
    print(f"\n{len(r.decisions)} decisions, {len(r.action_items)} action items, "
          f"{sum(e.accepted for e in r.edits)}/{len(r.edits)} edits accepted")
    for name, p in res.files.items():
        print(f"  {name}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
