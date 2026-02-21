from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(b)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        content = f.read()
    cfg = yaml.safe_load(content)
    cfg["_config_hash"] = sha256_bytes(content)
    cfg["_config_path"] = path
    return cfg


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


class Timer:
    def __init__(self):
        import time
        self.start_ns = time.perf_counter_ns()

    def elapsed_ms(self) -> float:
        import time
        return (time.perf_counter_ns() - self.start_ns) / 1e6
