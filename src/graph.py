# src/graph.py
from __future__ import annotations
"""LangGraph wiring for ingest, feature computation, policy, and provenance."""

from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

import math

from schemas import AdviceOutput, FrameInput, Advice
from utils import Timer, utc_now_iso
from lev import compute_lev
from policy_a4md_current import A4MDCurrentPolicy, A4MDPolicyState
from prov import build_prov_document_frame

# NEW (Paso 1B): optional post-hoc explanation node
from llm_explain import llm_explain_bullets


class AgentState(TypedDict, total=False):
    """Mutable state passed between graph nodes."""

    config: Dict[str, Any]
    frame: FrameInput
    policy_state: Dict[str, Any]
    decision: AdviceOutput
    evidence: List[Dict[str, Any]]
    timing: Dict[str, Dict[str, Any]]
    prov_doc: Dict[str, Any]


def _mark(state: AgentState, node: str, t: Timer, start: str) -> None:
    """Record timing metadata for a graph node."""
    timing = state.get("timing", {})
    timing[node] = {"start": start, "end": utc_now_iso(), "latency_ms": t.elapsed_ms()}
    state["timing"] = timing


def node_ingest(state: AgentState) -> AgentState:
    """Initialize per-trajectory policy state container if absent."""
    t = Timer()
    start = utc_now_iso()
    if "policy_state" not in state:
        state["policy_state"] = {}
    _mark(state, "ingest_frame", t, start)
    return state


def node_compute_lev(state: AgentState) -> AgentState:
    """Compute LEV when raw geometry is present and LEV is missing."""
    t = Timer()
    start = utc_now_iso()
    fr = state["frame"]
    if fr.features.lev is None and fr.points is not None and fr.atom_index_groups is not None:
        fr.features.lev = compute_lev(fr.points, fr.atom_index_groups)
    _mark(state, "compute_lev", t, start)
    return state


def _extract_lev_hits(evidence: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Pulls window_hits + window_min_hits from the most recent lev_check evidence.
    Returns (-1, -1) if not present.
    """
    window_hits = -1
    window_min_hits = -1

    for e in reversed(evidence):
        if "lev_check" in e:
            c = e["lev_check"]
            window_hits = int(c.get("stable_sum", -1))

            # H1 preferred key (int)
            if "stable_min_hits" in c and c["stable_min_hits"] is not None:
                window_min_hits = int(c["stable_min_hits"])
            else:
                # Back-compat: older float threshold stable_needed
                needed = c.get("stable_needed", -1)
                if isinstance(needed, (float, int)):
                    window_min_hits = int(math.ceil(float(needed)))
                else:
                    window_min_hits = -1
            break

    return window_hits, window_min_hits


def node_advisory(state: AgentState) -> AgentState:
    """Run the current policy update and emit normalized `AdviceOutput`."""
    t = Timer()
    start = utc_now_iso()
    cfg = state["config"]
    fr = state["frame"]

    pol = A4MDCurrentPolicy(cfg)

    if not state.get("policy_state"):
        st = pol.init_state()
    else:
        st = A4MDPolicyState(**state["policy_state"])

    # Default evidence
    evidence: List[Dict[str, Any]] = []

    if fr.features.lev is None:
        d_advice = "continue"
        d_conf = 0.5
        evidence = [{"fail_open": True, "reason": "LEV_unavailable"}]
        st.end_ev = st.end_ess = st.end_both = -1
    else:
        d = pol.update(st, fr.frame_id, fr.features.lev)
        d_advice, d_conf, evidence = d.advice, d.confidence, d.evidence

    state["policy_state"] = st.__dict__

    # Extract LEV rolling-window hits info (if present)
    window_hits, window_min_hits = _extract_lev_hits(evidence)

    out = AdviceOutput(
        advice=Advice.CONSIDER_EARLY_STOP if d_advice == "consider-early-stop" else Advice.CONTINUE,
        confidence=float(d_conf),
        rationale=[],
        policy_version=cfg["policy"]["policy_version"],
        graph_version=cfg["graph"]["graph_version"],
        config_hash=cfg["_config_hash"],
        traj_id=fr.traj_id,
        frame_id=fr.frame_id,
        window_size=int(cfg["a4md_current"]["window_lev"]),
        window_hits=window_hits,
        window_min_hits=window_min_hits,
        mode=str(cfg["a4md_current"]["decision_mode"]),
        fail_open=False,
        latency_ms=float(t.elapsed_ms()),
    )

    state["decision"] = out
    state["evidence"] = evidence
    _mark(state, "advisory_decision", t, start)
    return state


def _explain_impl(state: AgentState) -> AgentState:
    """Create deterministic rationale bullets from evidence and decision mode."""
    t = Timer()
    start = utc_now_iso()
    fr = state["frame"]
    d = state["decision"]
    evidence = state.get("evidence", [])

    decision_mode = str(state["config"]["a4md_current"]["decision_mode"])

    # ESS display rules (to match your latest desired behavior)
    has_ess_info = any(("ess_check" in e or "ess_termination" in e) for e in evidence)
    has_lev_term = any(("lev_termination" in e) for e in evidence)
    has_ess_term = any(("ess_termination" in e) for e in evidence)
    is_stop = (getattr(d.advice, "value", str(d.advice)) == "consider-early-stop")

    omit_ess = (decision_mode == "lev")
    suppress_ess = (decision_mode == "min" and is_stop and has_lev_term and not has_ess_term)

    bullets: List[str] = []
    bullets.append(f"- decision_mode=`{decision_mode}`")
    bullets.append(f"- frame={fr.frame_id}")
    if fr.features.lev is not None:
        bullets.append(f"- LEV={fr.features.lev:.6f}")

    # Evidence bullets
    for e in evidence:
        if "lev_check" in e:
            c = e["lev_check"]
            bullets.append(f"- LEV in_range={c['in_range']} range={c['range']}")
            stable_sum = c.get("stable_sum", None)

            # New key (H1): stable_min_hits (int)
            # Old key: stable_needed (float)
            needed = c.get("stable_min_hits", c.get("stable_needed", None))

            if isinstance(needed, float):
                needed_str = f"{needed:.2f}"
            else:
                needed_str = str(needed)

            bullets.append(f"- stable_sum={stable_sum} needed>={needed_str}")

        if "lev_termination" in e:
            bullets.append(f"- LEV termination at end_ev={e['lev_termination']['end_ev']}")

        # ESS details: conditional
        if "ess_check" in e:
            if not omit_ess and not suppress_ess:
                c = e["ess_check"]
                bullets.append(f"- ESS davg={c['davg']:.6f} threshold={c['threshold']:.6f}")

        if "ess_termination" in e:
            if not omit_ess and not suppress_ess:
                bullets.append(f"- ESS termination at end_ess={e['ess_termination']['end_ess']}")

        if "fail_open" in e:
            bullets.append(f"- Fail-open: {e}")

    # Add single summary line for ESS when we hide it
    if omit_ess and has_ess_info:
        bullets.append("- ESS check omitted (lev mode)")
    elif suppress_ess and has_ess_info:
        bullets.append("- ESS details suppressed (min mode; LEV triggered stop first)")

    state["decision"] = d.model_copy(update={"rationale": bullets})
    _mark(state, "explanation", t, start)
    return state


node_explanation = RunnableLambda(_explain_impl)


def node_llm_explanation(state: AgentState) -> AgentState:
    """
    Optional post-hoc explanation using an LLM (Paso 1B).
    This does NOT affect the decision. It only appends extra bullets to decision.rationale
    when enabled via config/env.
    """
    extra = llm_explain_bullets(state)  # returns None if disabled
    if not extra:
        return state

    t = Timer()
    start = utc_now_iso()

    d = state["decision"]
    bullets = list(d.rationale or [])
    bullets.extend(extra)

    state["decision"] = d.model_copy(update={"rationale": bullets})
    _mark(state, "llm_explanation", t, start)
    return state


def node_prov(state: AgentState) -> AgentState:
    """Materialize per-frame provenance document from current graph state."""
    t = Timer()
    start = utc_now_iso()
    state["prov_doc"] = build_prov_document_frame(
        frame=state["frame"],
        advice=state["decision"],
        timing=state.get("timing", {}),
        evidence=state.get("evidence", []),
        window_state={"policy_state": state.get("policy_state", {})},
    )
    _mark(state, "provenance_log", t, start)
    return state


def build_graph() -> Any:
    """Construct and compile-ready graph topology for the advisory pipeline."""
    g = StateGraph(AgentState)
    g.add_node("ingest_frame", node_ingest)
    g.add_node("compute_lev", node_compute_lev)
    g.add_node("advisory_decision", node_advisory)
    g.add_node("explanation", node_explanation)

    # NEW (Paso 1B)
    g.add_node("llm_explanation", node_llm_explanation)

    g.add_node("provenance_log", node_prov)

    g.set_entry_point("ingest_frame")
    g.add_edge("ingest_frame", "compute_lev")
    g.add_edge("compute_lev", "advisory_decision")
    g.add_edge("advisory_decision", "explanation")

    # NEW (Paso 1B): insert LLM explanation between explanation and provenance
    g.add_edge("explanation", "llm_explanation")
    g.add_edge("llm_explanation", "provenance_log")

    g.add_edge("provenance_log", END)
    return g
