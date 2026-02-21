from __future__ import annotations
"""Shared utility helpers for hashing, config loading, and timing."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import yaml


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-serializable object."""
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(b)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML and inject provenance metadata about its source and digest."""
    with open(path, "rb") as f:
        content = f.read()
    cfg = yaml.safe_load(content)
    cfg["_config_hash"] = sha256_bytes(content)
    cfg["_config_path"] = path
    return cfg


def ensure_dirs(*paths: str) -> None:
    """Create one or more directories if they do not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


class Timer:
    """Lightweight monotonic timer used for per-node latency metrics."""

    def __init__(self):
        import time
        self.start_ns = time.perf_counter_ns()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds since initialization."""
        import time
        return (time.perf_counter_ns() - self.start_ns) / 1e6
