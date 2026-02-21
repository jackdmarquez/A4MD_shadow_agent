from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Set

from schemas import FrameInput
from utils import ensure_dirs, load_yaml
from graph import build_graph


def _load_dotenv_if_present() -> None:
    """
    Load environment variables from a local .env file if python-dotenv is installed.
    Works both when run_shadow is imported (eval/run_demo.py) and when executed via CLI.

    If python-dotenv isn't installed, this is a no-op.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        # No python-dotenv or other dotenv issue => ignore gracefully
        return


def iter_frames_jsonl(path: str) -> Iterable[FrameInput]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield FrameInput.model_validate_json(line)


def append_jsonl(path: str, obj: Any) -> None:
    """
    Appends one JSON object per line.
    Uses default=str so datetime / enums won't break.
    """
    ensure_dirs(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def _get_checkpointer():
    """
    Prefer a persistent checkpointer if available; otherwise fallback to memory.

    Why: some LangGraph installs don't ship sqlite checkpointer module by default.
    """
    # 1) Try sqlite if available
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        Path("outputs").mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string("outputs/checkpoints.sqlite")
    except Exception:
        pass

    # 2) Fallback to in-memory checkpointer (always available in most installs)
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore

        return MemorySaver()
    except Exception:
        # 3) If even MemorySaver isn't available, run without checkpointer
        return None


def run_shadow(
    config_path: str,
    input_jsonl: str,
    *,
    log_everything: bool = False,
    log_all_stops: bool = False,
    stop_real: bool = False,
) -> None:
    """
    Runs the LangGraph shadow-mode advisory.

    Output files:
      - outputs/advice.jsonl
      - outputs/prov.jsonl

    Logging modes:
      - log_everything=True:
            logs ALL frames (continue + consider-early-stop)
      - log_everything=False (shadow strict):
            logs ONLY consider-early-stop

            default behavior in strict mode:
              * logs ONLY the FIRST consider-early-stop per trajectory
            override:
              * log_all_stops=True => logs every consider-early-stop frame

    stop_real:
      - If True, once a trajectory triggers consider-early-stop, subsequent frames
        for that traj_id are skipped entirely (no invoke, no logging).
    """
    # Ensure .env is loaded even when run_shadow is imported by eval/run_demo.py
    _load_dotenv_if_present()

    cfg = load_yaml(config_path)
    ensure_dirs("outputs", "reports", "data")

    advice_path = cfg["logging"]["advice_path"]
    prov_path = cfg["logging"]["prov_path"]

    # ALWAYS create/truncate output files so they exist even if no early-stops happen
    Path(advice_path).parent.mkdir(parents=True, exist_ok=True)
    Path(prov_path).parent.mkdir(parents=True, exist_ok=True)
    Path(advice_path).write_text("", encoding="utf-8")
    Path(prov_path).write_text("", encoding="utf-8")

    # Build and compile LangGraph app
    g = build_graph()
    saver = _get_checkpointer()
    if saver is None:
        app = g.compile()
    else:
        app = g.compile(checkpointer=saver)

    stopped_trajs: Set[str] = set()

    for fr in iter_frames_jsonl(input_jsonl):
        tid = fr.traj_id

        if stop_real and tid in stopped_trajs:
            continue

        state = {"frame": fr, "config": cfg}
        result = app.invoke(state, config={"configurable": {"thread_id": tid}})

        decision = result["decision"]
        prov_doc = result["prov_doc"]

        advice_value = decision.advice.value if hasattr(decision, "advice") else str(decision)

        if log_everything:
            append_jsonl(advice_path, decision.model_dump(mode="json"))
            append_jsonl(prov_path, prov_doc)
            if advice_value == "consider-early-stop":
                stopped_trajs.add(tid)
            continue

        if advice_value == "consider-early-stop":
            if (not log_all_stops) and (tid in stopped_trajs):
                # default: log only FIRST stop per traj
                pass
            else:
                append_jsonl(advice_path, decision.model_dump(mode="json"))
                append_jsonl(prov_path, prov_doc)

            stopped_trajs.add(tid)


if __name__ == "__main__":
    # Load .env as early as possible for CLI runs
    _load_dotenv_if_present()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--input", required=True)

    p.add_argument(
        "--log-everything",
        action="store_true",
        help="Log ALL frames (continue + early-stop). If omitted, logs ONLY consider-early-stop.",
    )
    p.add_argument(
        "--log-all-stops",
        action="store_true",
        help="In strict mode, log EVERY consider-early-stop frame (default logs only the FIRST per traj).",
    )
    p.add_argument(
        "--stop-real",
        action="store_true",
        help="After first consider-early-stop for a traj_id, skip the rest of its frames entirely.",
    )

    args = p.parse_args()

    run_shadow(
        args.config,
        args.input,
        log_everything=bool(args.log_everything),
        log_all_stops=bool(args.log_all_stops),
        stop_real=bool(args.stop_real),
    )
