from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from schemas import FrameInput, FrameFeatures
from utils import ensure_dirs


@dataclass
class SynthCfg:
    seed: int = 42
    n_traj: int = 10
    frames_per_traj: int = 200


def generate(out_path: str, cfg: SynthCfg) -> None:
    rng = np.random.default_rng(cfg.seed)
    ensure_dirs("data")

    with open(out_path, "w", encoding="utf-8") as f:
        for ti in range(cfg.n_traj):
            traj_id = f"traj-{ti:04d}"
            run_id = "synth-run-001"

            # Make LEV drift into range after some time (simulate stabilization)
            settle = int(rng.integers(low=30, high=120))
            for frame_id in range(cfg.frames_per_traj):
                if frame_id < settle:
                    lev = float(rng.normal(150.0, 20.0))
                else:
                    lev = float(rng.normal(50.0, 5.0))

                fr = FrameInput(
                    run_id=run_id,
                    traj_id=traj_id,
                    frame_id=frame_id,
                    features=FrameFeatures(lev=lev, ess=None),
                    meta={"synth_settle_frame": settle},
                )
                f.write(json.dumps(fr.model_dump(mode="json"), sort_keys=True) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synth_frames.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_traj", type=int, default=10)
    p.add_argument("--frames_per_traj", type=int, default=200)
    args = p.parse_args()

    generate(args.out, SynthCfg(seed=args.seed, n_traj=args.n_traj, frames_per_traj=args.frames_per_traj))
