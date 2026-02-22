# A4MD_shadow_agent

## How to run (end-to-end)

Run the workflow in this order:

1. Generate synthetic frames
2. Run the shadow agent
3. Run `prov_qa` on the generated provenance output

### 1) Generate synthetic frames

```bash
PYTHONPATH=src python tools/synthetic_frames.py --out data/synth_frames.jsonl
```

Optional flags:
- `--seed`
- `--n_traj`
- `--frames_per_traj`

### 2) Run shadow mode

```bash
PYTHONPATH=src python src/run_shadow.py --config configs/default.yaml --input data/synth_frames.jsonl
```

This writes outputs to:
- `outputs/advice.jsonl`
- `outputs/prov.jsonl`

### 3) Run provenance QA (`prov_qa`)

Example using the earliest stop event:

```bash
python tools/prov_qa.py --first-stop --question "Why did the agent recommend early stop?"
```

You can also target a specific trajectory/frame:

```bash
python tools/prov_qa.py --traj traj-0000 --frame 50 --question "What evidence supported this decision?"
```
