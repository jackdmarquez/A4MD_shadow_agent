from __future__ import annotations
"""Current A4MD policy implementation for LEV/ESS early-stop advisory."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math

import numpy as np


def n_effective_FTZ(x: np.ndarray) -> float:
    """
    Effective sample size via initial positive sequence (FTZ-style).
    """
    n = len(x)
    if n == 0:
        return 0.0

    xbar = np.mean(x)
    xnorm = x - xbar

    denom = float(np.sum(xnorm * xnorm))
    if denom == 0.0:
        return float(n)

    r: List[float] = []
    nc = 0
    for k in range(n - 1):
        r_k = float(np.sum(xnorm[0 : n - 1 - k] * xnorm[k + 1 : n]) / denom)
        r.append(r_k)
        if r_k < 0:
            nc = k - 1
            break

    ss = 0.0
    for k in range(max(0, nc)):
        ss += r[k]

    neff = n / (1.0 + 2.0 * ss)
    neff = min(neff, n)
    return float(neff)


@dataclass
class A4MDDecision:
    """Policy decision with trigger markers and evidence payload."""

    advice: str
    confidence: float
    end_ev: int
    end_ess: int
    end_both: int
    evidence: List[Dict[str, Any]]


@dataclass
class A4MDPolicyState:
    """Rolling policy state carried across frames in a trajectory."""

    n_frames_seen: int = 0
    end_ev: Optional[int] = None
    end_ess: Optional[int] = None
    end_both: Optional[int] = None
    ev_done: bool = False
    ess_done: bool = False
    stable_evs: Optional[List[int]] = None
    last_idx: int = 0
    evs_seen: Optional[List[float]] = None
    ess_by_frame: Optional[List[float]] = None
    x: Optional[List[int]] = None


class A4MDCurrentPolicy:
    """Stateful policy combining LEV and ESS criteria with configurable mode."""

    def __init__(self, cfg: Dict[str, Any]):
        """Read policy thresholds/window settings from configuration."""
        p = cfg["a4md_current"]
        self.window_lev = int(p["window_lev"])
        self.window_ess = int(p["window_ess"])
        self.range_min = float(p["range_min"])
        self.range_max = float(p["range_max"])
        self.range_th = float(p["range_th"])
        self.stable_th_lev = float(p["stable_th_lev"])
        self.var_th_ess = float(p["var_th_ess"])
        self.decision_mode = str(p.get("decision_mode", "min"))
        self.both_mode = str(p.get("both_mode", "last"))

        # Guardrails
        if self.window_lev < 1:
            self.window_lev = 1
        if self.window_ess < 1:
            self.window_ess = 1

    def init_state(self) -> A4MDPolicyState:
        """Create an initialized empty policy state."""
        return A4MDPolicyState(
            stable_evs=[0] * self.window_lev,
            last_idx=0,
            evs_seen=[],
            ess_by_frame=[],
            x=[],
        )

    def update(self, st: A4MDPolicyState, frame_id: int, ev: float) -> A4MDDecision:
        """Advance policy state for one frame and return advisory decision."""
        # Normalize nullable lists
        if st.stable_evs is None:
            st.stable_evs = [0] * self.window_lev
        if st.evs_seen is None:
            st.evs_seen = []
        if st.ess_by_frame is None:
            st.ess_by_frame = []
        if st.x is None:
            st.x = []

        st.n_frames_seen += 1
        current_frame_num = frame_id + 1  # 1-based counter

        # Sentinels
        if st.end_ev is None:
            st.end_ev = -1
        if st.end_ess is None:
            st.end_ess = -1
        if st.end_both is None:
            st.end_both = -1

        evidence: List[Dict[str, Any]] = []

        # Track EV series
        st.evs_seen.append(float(ev))
        st.x.append(frame_id)

        # -------------------------
        # LEV window stability check
        # -------------------------
        if (st.end_ev == -1) and (current_frame_num >= self.window_lev) and (not st.ev_done):
            in_range = (ev >= (self.range_min - self.range_th)) and (ev <= (self.range_max + self.range_th))
            st.stable_evs[st.last_idx] = 1 if in_range else 0

            stable_sum = int(np.sum(st.stable_evs))
            stable_needed_float = (self.stable_th_lev / 100.0) * float(self.window_lev)
            stable_min_hits = int(math.ceil(stable_needed_float))

            evidence.append(
                {
                    "lev_check": {
                        "ev": float(ev),
                        "in_range": bool(in_range),
                        "range": [self.range_min - self.range_th, self.range_max + self.range_th],
                        "stable_sum": stable_sum,
                        # Keep both keys for backwards compatibility:
                        "stable_needed": float(stable_needed_float),
                        "stable_min_hits": int(stable_min_hits),
                        "window_lev": int(self.window_lev),
                    }
                }
            )

            if stable_sum >= stable_min_hits:
                st.end_ev = current_frame_num
                st.ev_done = True
                evidence.append({"lev_termination": {"end_ev": st.end_ev}})

            st.last_idx = (st.last_idx + 1) % self.window_lev

        # -------------------------
        # ESS computation + stability check
        # -------------------------
        lo = max(0, current_frame_num - self.window_ess)
        ess_val = n_effective_FTZ(np.array(st.evs_seen[lo:current_frame_num], dtype=float))
        st.ess_by_frame.append(float(ess_val))

        # Gate ESS termination in MIN mode so it cannot "win" before LEV is stable.
        # This fixes the "min stops at frame 19" behavior you saw.
        ess_gate_ok = True
        if self.decision_mode == "min" and (not st.ev_done):
            ess_gate_ok = False

        if (
            ess_gate_ok
            and (st.end_ess == -1)
            and (current_frame_num >= self.window_ess)
            and (not st.ess_done)
        ):
            ess_window = np.array(st.ess_by_frame[-self.window_ess:], dtype=float)
            dy = np.diff(ess_window)
            davg = float(np.average(dy)) if len(dy) > 0 else 0.0

            # var_th_ess is treated as a percent; convert to absolute delta threshold
            th = self.var_th_ess / 100.0

            evidence.append({"ess_check": {"davg": davg, "threshold": th, "window_ess": self.window_ess}})

            # Stability should mean "changes are small", not "changes are big"
            if abs(davg) <= th:
                st.end_ess = current_frame_num
                st.ess_done = True
                evidence.append({"ess_termination": {"end_ess": st.end_ess}})
        elif (not ess_gate_ok) and (current_frame_num >= self.window_ess) and (not st.ess_done):
            # Optional breadcrumb for debugging/explanations
            evidence.append({"ess_check_skipped": {"reason": "min_mode_waiting_for_lev_done"}})

        # -------------------------
        # BOTH bookkeeping
        # -------------------------
        if st.ev_done and st.ess_done:
            if self.both_mode == "first":
                if st.end_both == -1:
                    st.end_both = current_frame_num
            else:
                st.end_both = current_frame_num

        # -------------------------
        # Decision trigger
        # -------------------------
        end_ev = st.end_ev if st.end_ev != -1 else 10**12
        end_ess = st.end_ess if st.end_ess != -1 else 10**12
        end_both = st.end_both if st.end_both != -1 else 10**12

        if self.decision_mode == "lev":
            trigger = (st.end_ev != -1) and (current_frame_num >= st.end_ev)
            why = "lev"
        elif self.decision_mode == "ess":
            trigger = (st.end_ess != -1) and (current_frame_num >= st.end_ess)
            why = "ess"
        elif self.decision_mode == "both":
            trigger = (st.end_both != -1) and (current_frame_num >= st.end_both)
            why = "both"
        else:
            m = min(end_ev, end_ess, end_both)
            trigger = current_frame_num >= m
            why = "min"

        if trigger:
            evidence.append({"decision": {"mode": self.decision_mode, "why": why}})
            return A4MDDecision("consider-early-stop", 0.85, st.end_ev, st.end_ess, st.end_both, evidence)

        return A4MDDecision("continue", 0.70, st.end_ev, st.end_ess, st.end_both, evidence)
