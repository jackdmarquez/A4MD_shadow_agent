# src/llm_explain.py
from __future__ import annotations
"""Optional LLM-based rationale augmentation for advisory outputs."""

import os
from typing import Any, Dict, List, Optional


def _env_truthy(name: str, default: str = "0") -> bool:
    """Interpret environment variable values with permissive truthy parsing."""
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def llm_explain_enabled(cfg: Dict[str, Any]) -> bool:
    """Return whether LLM explanations are enabled via env/config."""
    # priority: env var -> config
    if _env_truthy("A4MD_LLM_EXPLAIN", "0"):
        return True
    llm_cfg = cfg.get("llm_explain", {}) if isinstance(cfg, dict) else {}
    return bool(llm_cfg.get("enabled", False))


def _get_openai_chat_model(cfg: Dict[str, Any]):
    """
    Returns a LangChain chat model if available; otherwise None.

    We intentionally keep this optional so the repo runs without any LLM deps/keys.
    """
    # Try modern provider package first
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:
        ChatOpenAI = None  # type: ignore

    if ChatOpenAI is None:
        return None

    # Require an API key to actually call; otherwise skip (no crash)
    if not os.getenv("OPENAI_API_KEY"):
        return None

    llm_cfg = cfg.get("llm_explain", {}) if isinstance(cfg, dict) else {}

    model = (
        os.getenv("A4MD_LLM_MODEL")
        or llm_cfg.get("model")
        or os.getenv("OPENAI_MODEL")
        or ""
    ).strip()

    # If user didn't specify, we skip instead of guessing a model name.
    if not model:
        return None

    temperature = float(llm_cfg.get("temperature", 0.0))

    # Keep it cheap/fast by default
    return ChatOpenAI(model=model, temperature=temperature)


def llm_explain_bullets(state: Dict[str, Any]) -> Optional[List[str]]:
    """
    Returns extra rationale bullets (strings) to append to AdviceOutput.rationale,
    or None if LLM explain is disabled/unavailable.

    Never raises: failures degrade gracefully to "skipped" (or an error bullet when enabled).
    """
    cfg = state.get("config", {})
    if not llm_explain_enabled(cfg):
        return None

    llm = _get_openai_chat_model(cfg)
    if llm is None:
        # enabled, but no provider/keys/model configured
        return ["- LLM explanation skipped (no provider/model configured)"]

    # Build a compact, safe summary for the LLM (avoid huge payloads)
    fr = state.get("frame")
    d = state.get("decision")
    evidence = state.get("evidence", [])

    payload = {
        "traj_id": getattr(fr, "traj_id", None),
        "frame_id": getattr(fr, "frame_id", None),
        "decision_mode": getattr(d, "mode", None) if d is not None else None,
        "advice": getattr(getattr(d, "advice", None), "value", None) if d is not None else None,
        "rationale": getattr(d, "rationale", None) if d is not None else None,
        "evidence": evidence,
    }

    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore

        sys = (
            "You are an assistant explaining an agent's early-stop decision in a scientific workflow.\n"
            "Rules:\n"
            "- Use ONLY the provided payload; do not assume missing info.\n"
            "- Output 2-4 short bullet points.\n"
            "- Keep it technical but readable (no fluff).\n"
            "- If there is not enough info, say what is missing.\n"
        )
        user = (
            "Explain why the agent produced its advice for this frame.\n"
            "Focus on thresholds, window/hits, and what triggered (LEV vs ESS vs both).\n"
            f"Payload:\n{payload}"
        )

        resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        text = (getattr(resp, "content", "") or "").strip()
        if not text:
            return ["- LLM explanation produced empty output"]

        # Normalize: ensure it becomes bullets appended to rationale
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets: List[str] = []
        for ln in lines:
            if ln.startswith("-"):
                bullets.append(ln)
            else:
                bullets.append(f"- {ln}")

        return bullets[:6]

    except Exception as e:
        return [f"- LLM explanation error: {type(e).__name__}: {e}"]
