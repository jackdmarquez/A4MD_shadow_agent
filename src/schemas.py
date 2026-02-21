from __future__ import annotations
"""Pydantic schemas exchanged between ingestion, policy, and logging layers."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Advice(str, Enum):
    """Canonical advisory outcomes produced by the policy graph."""

    CONTINUE = "continue"
    CONSIDER_EARLY_STOP = "consider-early-stop"


class FrameFeatures(BaseModel):
    """Per-frame derived or precomputed numerical features."""

    model_config = ConfigDict(extra="forbid")
    lev: Optional[float] = None
    ess: Optional[float] = None


class FrameInput(BaseModel):
    """Single frame of an MD trajectory with optional geometric payload."""

    model_config = ConfigDict(extra="allow")
    run_id: str
    traj_id: str
    frame_id: int = Field(..., ge=0)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    points: Optional[Any] = None
    atom_index_groups: Optional[Any] = None
    features: FrameFeatures = Field(default_factory=FrameFeatures)
    meta: Dict[str, Any] = Field(default_factory=dict)


class AdviceOutput(BaseModel):
    """Decision artifact emitted by advisory logic and persisted to JSONL."""

    model_config = ConfigDict(extra="forbid")
    advice: Advice
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: List[str]
    policy_version: str
    graph_version: str
    config_hash: str
    traj_id: str
    frame_id: int
    window_size: int
    window_hits: int
    window_min_hits: int
    mode: str
    fail_open: bool = False
    latency_ms: float = 0.0
