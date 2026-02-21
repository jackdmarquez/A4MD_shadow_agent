from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env early
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_traj_frame(doc: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    tid = doc.get("traj_id")
    fid = doc.get("frame_id")
    if isinstance(tid, str) and isinstance(fid, int):
        return tid, fid
    if isinstance(tid, str) and isinstance(fid, str) and fid.isdigit():
        return tid, int(fid)

    ents = doc.get("entities", [])
    if isinstance(ents, list):
        for e in ents:
            if not isinstance(e, dict):
                continue
            if e.get("type") != "FrameInput":
                continue
            attrs = e.get("attrs", {})
            if not isinstance(attrs, dict):
                continue
            tid2 = attrs.get("traj_id")
            fid2 = attrs.get("frame_id")
            if isinstance(tid2, str) and isinstance(fid2, int):
                return tid2, fid2
            if isinstance(tid2, str) and isinstance(fid2, str) and fid2.isdigit():
                return tid2, int(fid2)

    return None, None


def _bundle_id(doc: Dict[str, Any]) -> str:
    b = doc.get("bundle_id")
    return b if isinstance(b, str) else ""


def list_available_pairs(docs: List[Dict[str, Any]]) -> List[Tuple[str, int, str]]:
    pairs: List[Tuple[str, int, str]] = []
    for d in docs:
        tid, fid = _extract_traj_frame(d)
        if isinstance(tid, str) and isinstance(fid, int):
            pairs.append((tid, fid, _bundle_id(d)))
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs


def _safe_json(obj: Any, max_chars: int = 14000) -> str:
    s = json.dumps(obj, indent=2, sort_keys=True, default=str)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 200] + "\n...\n(TRUNCATED)\n"


def _find_doc_strict(docs: List[Dict[str, Any]], traj_id: str, frame_id: int) -> Dict[str, Any]:
    for d in docs:
        tid, fid = _extract_traj_frame(d)
        if tid == traj_id and fid == frame_id:
            return d
    raise KeyError(f"target_not_found traj={traj_id} frame={frame_id}")


def _extract_frame_entity(doc: Dict[str, Any]) -> Dict[str, Any]:
    ents = doc.get("entities", [])
    if not isinstance(ents, list):
        return {}
    for e in ents:
        if isinstance(e, dict) and e.get("type") == "FrameInput":
            return e
    return {}


def _extract_advice_entity(doc: Dict[str, Any]) -> Dict[str, Any]:
    ents = doc.get("entities", [])
    if not isinstance(ents, list):
        return {}
    for e in ents:
        if isinstance(e, dict) and e.get("type") == "AdviceOutput":
            return e
    return {}


def _extract_advisory_evidence(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    acts = doc.get("activities", [])
    if not isinstance(acts, list):
        return []
    for a in acts:
        if not isinstance(a, dict):
            continue
        if a.get("type") != "advisory_decision":
            continue
        attrs = a.get("attrs", {})
        if isinstance(attrs, dict):
            ev = attrs.get("evidence", [])
            if isinstance(ev, list):
                return [x for x in ev if isinstance(x, dict)]
    return []


def _pick(d: Any, keys: List[str]) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def summarize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    tid, fid = _extract_traj_frame(doc)

    frame_ent = _extract_frame_entity(doc)
    frame_attrs = frame_ent.get("attrs", {}) if isinstance(frame_ent, dict) else {}
    if not isinstance(frame_attrs, dict):
        frame_attrs = {}

    advice_ent = _extract_advice_entity(doc)
    advice_attrs = advice_ent.get("attrs", {}) if isinstance(advice_ent, dict) else {}
    if not isinstance(advice_attrs, dict):
        advice_attrs = {}

    evidence = _extract_advisory_evidence(doc)

    lev_check = None
    ess_check = None
    lev_term = None
    ess_term = None
    decision = None
    for e in evidence:
        if "lev_check" in e:
            lev_check = e["lev_check"]
        if "ess_check" in e:
            ess_check = e["ess_check"]
        if "lev_termination" in e:
            lev_term = e["lev_termination"]
        if "ess_termination" in e:
            ess_term = e["ess_termination"]
        if "decision" in e:
            decision = e["decision"]

    features = frame_attrs.get("features", {}) if isinstance(frame_attrs.get("features"), dict) else {}
    lev_val = _pick(lev_check, ["ev"]) or _pick(features, ["lev"])

    lev_in_range = _pick(lev_check, ["in_range"])
    stable_sum = _pick(lev_check, ["stable_sum"])
    stable_min_hits = _pick(lev_check, ["stable_min_hits"])
    window_lev = _pick(lev_check, ["window_lev"])
    end_ev = _pick(lev_term, ["end_ev"])

    trigger = _pick(decision, ["why", "mode"]) or advice_attrs.get("mode")

    key_evidence = {
        "lev": lev_val,
        "lev_in_range": lev_in_range,
        "stable_sum": stable_sum,
        "stable_min_hits": stable_min_hits,
        "window_lev": window_lev,
        "end_ev": end_ev,
        "trigger": trigger,
    }

    return {
        "resolved_target": {"traj_id": tid, "frame_id": fid, "bundle_id": _bundle_id(doc)},
        "frame": {
            "run_id": frame_attrs.get("run_id"),
            "traj_id": frame_attrs.get("traj_id"),
            "frame_id": frame_attrs.get("frame_id"),
            "features": frame_attrs.get("features"),
            "meta": frame_attrs.get("meta"),
        },
        "advice": {
            "advice": advice_attrs.get("advice"),
            "confidence": advice_attrs.get("confidence"),
            "mode": advice_attrs.get("mode"),
            "policy_version": advice_attrs.get("policy_version"),
            "graph_version": advice_attrs.get("graph_version"),
            "config_hash": advice_attrs.get("config_hash"),
            "rationale": advice_attrs.get("rationale"),
        },
        "evidence": {
            "lev_check": lev_check,
            "ess_check": ess_check,
            "lev_termination": lev_term,
            "ess_termination": ess_term,
            "decision": decision,
        },
        "key_evidence": key_evidence,
    }


def _first_stop_targets_from_advice(advice_docs: List[Dict[str, Any]]) -> Dict[str, int]:
    first: Dict[str, int] = {}
    for a in advice_docs:
        if not isinstance(a, dict):
            continue
        if a.get("advice") != "consider-early-stop":
            continue
        tid = a.get("traj_id")
        fid = a.get("frame_id")
        if not isinstance(tid, str) or not isinstance(fid, int):
            continue
        if tid not in first or fid < first[tid]:
            first[tid] = fid
    return first


def _pick_global_earliest(first_by_traj: Dict[str, int]) -> Tuple[Optional[str], Optional[int]]:
    if not first_by_traj:
        return None, None
    items = sorted(first_by_traj.items(), key=lambda kv: (kv[1], kv[0]))
    return items[0][0], items[0][1]


def build_context(
    prov_docs: List[Dict[str, Any]],
    traj_id: Optional[str],
    frame_id: Optional[int],
    last_k: int,
    strict: bool,
    include_full_doc: bool,
) -> str:
    if not prov_docs:
        return "No provenance documents found."

    selected_docs: List[Dict[str, Any]] = []

    if traj_id is not None and frame_id is not None:
        if strict:
            d = _find_doc_strict(prov_docs, traj_id, frame_id)
            selected_docs = [d]
        else:
            try:
                d = _find_doc_strict(prov_docs, traj_id, frame_id)
                selected_docs = [d]
            except Exception:
                selected_docs = prov_docs[-last_k:]
    else:
        selected_docs = prov_docs[-last_k:]

    summary = summarize_doc(selected_docs[0])

    payload: Dict[str, Any] = {
        "selection": {
            "requested": {"traj_id": traj_id, "frame_id": frame_id, "last_k": last_k, "strict": strict},
            "resolved_target": summary.get("resolved_target"),
            "n_docs_total": len(prov_docs),
            "n_docs_selected": len(selected_docs),
        },
        "summary": summary,
    }

    if include_full_doc:
        payload["prov_doc_full"] = selected_docs[0]

    return _safe_json(payload)


def get_chat_model() -> Optional[Any]:
    provider = os.getenv("SHADOW_LLM_PROVIDER", "auto").strip().lower()

    if provider in ("auto", "openai"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "OpenAI selected but langchain_openai is not installed. "
                    "Install it or switch SHADOW_LLM_PROVIDER=ollama/none."
                ) from e

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
            temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
            return ChatOpenAI(model=model, temperature=temp)

        if provider == "openai":
            raise RuntimeError("SHADOW_LLM_PROVIDER=openai pero OPENAI_API_KEY está vacío.")

    if provider in ("auto", "ollama"):
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as e:
            if provider == "ollama":
                raise RuntimeError(
                    "SHADOW_LLM_PROVIDER=ollama pero no tienes langchain-ollama instalado. "
                    "Instala: pip install -U langchain-ollama"
                ) from e
            return None

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "llama3.1:latest").strip()
        temp = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        return ChatOllama(model=model, base_url=base_url, temperature=temp)

    return None


def deterministic_answer(question: str, context: str) -> str:
    return (
        "Deterministic fallback (no LLM configured).\n"
        f"Question: {question}\n\n"
        "Context (JSON):\n"
        f"{context}"
    )


def answer_with_llm(question: str, context: str) -> str:
    llm = get_chat_model()
    if llm is None:
        return deterministic_answer(question, context)

    system = (
        "You are a provenance Q&A assistant for an agentic scientific workflow.\n"
        "You MUST answer ONLY about selection.resolved_target.\n"
        "Use summary.key_evidence FIRST for numeric fields.\n"
        "\n"
        "Output MUST match exactly:\n"
        "\n"
        "Selected target: traj=<traj_id> frame=<frame_id>\n"
        "Cause:\n"
        "- advice: <advice>\n"
        "- mode: <mode>\n"
        "- trigger: <lev|ess|both|min>\n"
        "Key evidence:\n"
        "- lev: <number or null>\n"
        "- lev_in_range: <true/false/null>\n"
        "- stable_sum: <int or null>\n"
        "- stable_min_hits: <int or null>\n"
        "- window_lev: <int or null>\n"
        "- end_ev: <int or null>\n"
        "Notes:\n"
        "- <1 sentence that cites the stable_sum/min_hits + in_range>\n"
        "\n"
        "Rules:\n"
        "- If a field is missing, write null.\n"
        "- trigger MUST be exactly one of: lev, ess, both, min.\n"
        "- Notes must NOT say 'high' or vague; it must say the condition.\n"
        "- Do NOT reference any other trajectory/frame.\n"
    )

    messages = [
        ("system", system),
        ("human", f"QUESTION:\n{question}\n\nCONTEXT JSON:\n{context}"),
    ]

    try:
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        return (
            "LLM call failed; falling back to deterministic mode.\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            + deterministic_answer(question, context)
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prov", default="outputs/prov.jsonl", help="Path to provenance JSONL.")
    p.add_argument("--advice", default="outputs/advice.jsonl", help="Path to advice JSONL (for --first-stop).")
    p.add_argument("--question", default=None, help="Natural language question about the provenance.")
    p.add_argument("--traj", default=None, help="Optional traj_id (e.g., traj-0003).")
    p.add_argument("--frame", type=int, default=None, help="Optional frame_id (e.g., 45).")
    p.add_argument("--first-stop", action="store_true", help="Auto-select the earliest consider-early-stop target.")
    p.add_argument("--last-k", type=int, default=3, help="Docs to include when not using exact traj+frame.")
    p.add_argument("--list", action="store_true", help="List available (traj_id, frame_id) pairs and exit.")
    p.add_argument("--strict", action="store_true", help="Require exact traj+frame match; otherwise exit.")
    p.add_argument("--include-full-doc", action="store_true", help="Include full prov doc in context JSON (bigger).")
    args = p.parse_args()

    prov_docs = read_jsonl(args.prov)

    if args.list:
        pairs = list_available_pairs(prov_docs)
        if not pairs:
            print("No (traj_id, frame_id) pairs detected. Check prov schema.")
            return
        for tid, fid, bid in pairs:
            print(f"{tid}\t{fid}\t{bid}")
        return

    if not args.question:
        raise SystemExit("Missing --question (or use --list).")

    traj_id = args.traj
    frame_id = args.frame

    if traj_id is None and frame_id is None and args.first_stop:
        advice_docs = read_jsonl(args.advice)
        first_by_traj = _first_stop_targets_from_advice(advice_docs)
        t, f = _pick_global_earliest(first_by_traj)
        if t is None or f is None:
            raise SystemExit("No consider-early-stop entries found in advice.jsonl.")
        traj_id, frame_id = t, f

    strict = bool(args.strict)
    include_full_doc = bool(args.include_full_doc)

    if traj_id is not None and frame_id is not None:
        strict = True

    try:
        ctx = build_context(
            prov_docs,
            traj_id=traj_id,
            frame_id=frame_id,
            last_k=args.last_k,
            strict=strict,
            include_full_doc=include_full_doc,
        )
    except KeyError as e:
        print(f"ERROR: {e}")
        print("Available pairs:")
        for tid, fid, bid in list_available_pairs(prov_docs):
            print(f"  - {tid}:{fid} ({bid})")
        raise SystemExit(2)

    ans = answer_with_llm(args.question, ctx)
    print(ans)


if __name__ == "__main__":
    main()
