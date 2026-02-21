from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Advice(str, Enum):
    CONTINUE = "continue"
    CONSIDER_EARLY_STOP = "consider-early-stop"


class FrameFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lev: Optional[float] = None
    ess: Optional[float] = None


class FrameInput(BaseModel):
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
