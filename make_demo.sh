#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH=src

mkdir -p data outputs reports

python tools/synthetic_frames.py --out data/synth_frames.jsonl --seed 42 --n_traj 10 --frames_per_traj 200
python eval/run_demo.py --config configs/default.yaml --input data/synth_frames.jsonl --report reports/metrics.md

echo "Done."
ls -lh outputs/advice.jsonl outputs/prov.jsonl reports/metrics.md
