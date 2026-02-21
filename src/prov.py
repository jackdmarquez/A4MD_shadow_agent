from __future__ import annotations
"""Helpers for building simplified provenance documents per processed frame."""
from typing import Any, Dict, List

from schemas import AdviceOutput, FrameInput
from utils import sha256_json, utc_now_iso


def build_prov_document_frame(
    frame: FrameInput,
    advice: AdviceOutput,
    timing: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    window_state: Any,
) -> Dict[str, Any]:
    """Build a single provenance bundle linking input, state, and decision."""
    frame_hash = sha256_json({
        "run_id": frame.run_id,
        "traj_id": frame.traj_id,
        "frame_id": frame.frame_id,
        "features": frame.features.model_dump(),
        "meta": frame.meta,
    })
    advice_hash = sha256_json(advice.model_dump())

    return {
        "schema": "prov-simplified@0.3",
        "bundle_id": f"prov:{frame.run_id}:{frame.traj_id}:{frame.frame_id}:{frame_hash}",
        "generated_at": utc_now_iso(),
        "entities": [
            {
                "id": f"ent:frame:{frame_hash}",
                "type": "FrameInput",
                "time": frame.timestamp_utc.isoformat(),
                "attrs": {
                    "run_id": frame.run_id,
                    "traj_id": frame.traj_id,
                    "frame_id": frame.frame_id,
                    "features": frame.features.model_dump(),
                    "meta": frame.meta,
                },
            },
            {
                "id": f"ent:state:{frame_hash}",
                "type": "PolicyState",
                "time": utc_now_iso(),
                "attrs": window_state,
            },
            {
                "id": f"ent:advice:{advice_hash}",
                "type": "AdviceOutput",
                "time": utc_now_iso(),
                "attrs": advice.model_dump(),
            },
        ],
        "agents": [
            {
                "id": f"agent:policy:{advice.policy_version}",
                "type": "Policy",
                "attrs": {
                    "policy_version": advice.policy_version,
                    "graph_version": advice.graph_version,
                    "config_hash": advice.config_hash,
                },
            }
        ],
        "activities": [
            {
                "id": f"act:compute_lev:{frame_hash}",
                "type": "compute_lev",
                "start": timing.get("compute_lev", {}).get("start"),
                "end": timing.get("compute_lev", {}).get("end"),
                "attrs": {"latency_ms": timing.get("compute_lev", {}).get("latency_ms")},
            },
            {
                "id": f"act:advisory_decision:{frame_hash}",
                "type": "advisory_decision",
                "start": timing.get("advisory_decision", {}).get("start"),
                "end": timing.get("advisory_decision", {}).get("end"),
                "attrs": {
                    "latency_ms": timing.get("advisory_decision", {}).get("latency_ms"),
                    "evidence": evidence,
                },
            },
        ],
        "relations": {
            "wasDerivedFrom": [
                {"generatedEntity": f"ent:advice:{advice_hash}", "usedEntity": f"ent:frame:{frame_hash}"},
                {"generatedEntity": f"ent:advice:{advice_hash}", "usedEntity": f"ent:state:{frame_hash}"},
            ]
        },
    }
