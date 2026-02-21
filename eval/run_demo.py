# eval/run_demo.py
from __future__ import annotations
"""Run a complete shadow demo and emit a small markdown metrics report."""

import json
import sys
from pathlib import Path

from collections import defaultdict

# Ensure src/ is importable when running: python eval/run_demo.py
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_shadow import run_shadow  # noqa: E402
from utils import ensure_dirs, load_yaml  # noqa: E402


def read_jsonl(path: str):
    """Load JSONL records from disk; return an empty list when missing."""
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except FileNotFoundError:
        return []
    return out


def main(config_path: str, input_path: str, report_path: str) -> None:
    """Execute shadow run and summarize first stop frame per trajectory."""
    ensure_dirs("reports")
    cfg = load_yaml(config_path)

    run_shadow(config_path, input_path)

    advice = read_jsonl(cfg["logging"]["advice_path"])

    # simple metrics: first stop frame per trajectory (shadow)
    first_by_traj = {}
    for a in advice:
        tid = a["traj_id"]
        if tid not in first_by_traj:
            first_by_traj[tid] = a["frame_id"]

    md = []
    md.append("# Shadow demo report\n")
    md.append(f"- config_hash: `{cfg['_config_hash']}`")
    md.append(f"- policy_version: `{cfg['policy']['policy_version']}`\n")
    md.append("## First `consider-early-stop` frame per trajectory\n")
    md.append("| traj_id | first_stop_frame |\n|---|---:|\n")
    for tid in sorted(first_by_traj.keys()):
        md.append(f"| {tid} | {first_by_traj[tid]} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--input", default="data/synth_frames.jsonl")
    p.add_argument("--report", default="reports/metrics.md")
    args = p.parse_args()
    main(args.config, args.input, args.report)
