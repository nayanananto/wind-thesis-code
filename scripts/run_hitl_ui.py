from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.hitl.standalone_pipeline import StandaloneHITLPipeline  # noqa: E402
from app.hitl.semantic_review import load_semantic_metadata  # noqa: E402


DEFAULT_HOURLY_METADATA = PROJECT_ROOT / "data" / "metadata" / "kbos_hourly_hitl_semantic_build.json"
DEFAULT_5MIN_METADATA = PROJECT_ROOT / "data" / "metadata" / "kbos_5min_phase_semantic_build.json"
STATIC_DIR = PROJECT_ROOT / "app" / "hitl" / "ui_static"
DEFAULT_NUMERIC_MODEL = PROJECT_ROOT / "artifacts" / "hitl_numeric_lstm" / "kbos_hourly_v1"
DEFAULT_PHASE_MODEL = PROJECT_ROOT / "artifacts" / "hitl_phase_gru" / "kbos_hourly_v1"
DEFAULT_LOCAL_LIVE_HISTORY = PROJECT_ROOT / "data" / "live" / "semantic_states" / "KBOS_live_semantic_history.csv"
DEFAULT_LOCAL_LIVE_STATE = PROJECT_ROOT / "data" / "live" / "semantic_states" / "KBOS_live_semantic_state.json"
DEFAULT_LOCAL_LIVE_RAW = PROJECT_ROOT / "data" / "live" / "aviationweather_metar" / "KBOS" / "metar_live.csv"
DEFAULT_LIVE_PACKAGE = PROJECT_ROOT.parent / "wind_live_hitl_github_package"
DEFAULT_PACKAGE_LIVE_HISTORY = DEFAULT_LIVE_PACKAGE / "data" / "live" / "semantic_states" / "KBOS_live_semantic_history.csv"
DEFAULT_PACKAGE_LIVE_STATE = DEFAULT_LIVE_PACKAGE / "data" / "live" / "semantic_states" / "KBOS_live_semantic_state.json"
DEFAULT_PACKAGE_LIVE_RAW = DEFAULT_LIVE_PACKAGE / "data" / "live" / "aviationweather_metar" / "KBOS" / "metar_live.csv"


def _default_live_history_path() -> Path:
    return DEFAULT_PACKAGE_LIVE_HISTORY if DEFAULT_PACKAGE_LIVE_HISTORY.exists() else DEFAULT_LOCAL_LIVE_HISTORY


def _default_live_state_path() -> Path:
    return DEFAULT_PACKAGE_LIVE_STATE if DEFAULT_PACKAGE_LIVE_STATE.exists() else DEFAULT_LOCAL_LIVE_STATE


def _default_live_raw_path() -> Path:
    return DEFAULT_PACKAGE_LIVE_RAW if DEFAULT_PACKAGE_LIVE_RAW.exists() else DEFAULT_LOCAL_LIVE_RAW


def _default_metadata_path() -> Path:
    return DEFAULT_5MIN_METADATA if DEFAULT_5MIN_METADATA.exists() else DEFAULT_HOURLY_METADATA


def _sync_live_package_on_load() -> dict[str, Any]:
    """Fetch latest live GitHub data when the browser UI asks for defaults.

    The command is intentionally non-destructive. If the local live package has
    uncommitted changes, `git pull --ff-only` fails instead of overwriting them.
    """

    if not DEFAULT_LIVE_PACKAGE.exists():
        return {
            "attempted": False,
            "ok": False,
            "status": "missing_package",
            "message": f"Live package not found: {DEFAULT_LIVE_PACKAGE}",
        }
    if not (DEFAULT_LIVE_PACKAGE / ".git").exists():
        return {
            "attempted": False,
            "ok": False,
            "status": "not_git_repo",
            "message": f"Live package is not a git repo: {DEFAULT_LIVE_PACKAGE}",
        }

    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={DEFAULT_LIVE_PACKAGE.as_posix()}", "pull", "--ff-only"],
            cwd=DEFAULT_LIVE_PACKAGE,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "status": "error",
            "message": str(exc),
        }

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if len(output) > 800:
        output = output[-800:]
    return {
        "attempted": True,
        "ok": completed.returncode == 0,
        "status": "pulled" if completed.returncode == 0 else "pull_failed",
        "message": output or ("Already up to date." if completed.returncode == 0 else "git pull failed"),
    }


def _refresh_hourly_live_semantic_state() -> dict[str, Any]:
    if DEFAULT_PACKAGE_LIVE_HISTORY.exists() and DEFAULT_PACKAGE_LIVE_STATE.exists():
        return {
            "attempted": False,
            "ok": True,
            "status": "using_package_live_state",
            "message": "Using the synced GitHub live semantic state as the HITL live phase source.",
        }

    if not DEFAULT_HOURLY_METADATA.exists():
        return {
            "attempted": False,
            "ok": False,
            "status": "missing_hourly_metadata",
            "message": f"Hourly metadata not found: {DEFAULT_HOURLY_METADATA}",
        }

    live_raw = _default_live_raw_path()
    if not live_raw.exists():
        return {
            "attempted": False,
            "ok": False,
            "status": "missing_live_raw",
            "message": f"Live raw file not found: {live_raw}",
        }

    output_dir = PROJECT_ROOT / "data" / "live" / "semantic_states"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_live_semantic_state.py"),
        "--live_csv",
        str(live_raw),
        "--metadata",
        str(DEFAULT_HOURLY_METADATA),
        "--station",
        "KBOS",
        "--live_window_rows",
        "6",
        "--top_k",
        "5",
        "--phase_horizon_steps",
        "1",
        "--output_dir",
        str(output_dir),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "status": "error",
            "message": str(exc),
        }

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if len(output) > 800:
        output = output[-800:]
    return {
        "attempted": True,
        "ok": completed.returncode == 0,
        "status": "refreshed" if completed.returncode == 0 else "refresh_failed",
        "message": output or ("Live semantic state refreshed." if completed.returncode == 0 else "refresh failed"),
    }


class HITLUIRequest(BaseModel):
    question: str = Field(..., min_length=1)
    thread_id: str = "hitl_ui_default"
    metadata: str | None = None
    window_id: str | None = None
    latest: bool = True
    top_k: int = 5
    enable_llm: bool = False
    enable_agent_graph: bool = False
    enable_rag: bool = True
    confidence_threshold: float = 0.45
    phase_tokens: list[int] | None = None
    phase_history_length: int = 2
    phase_horizon_steps: int = 1
    phase_top_k: int = 3
    phase_analog_k: int = 5
    phase_min_support: int = 5
    live_raw_path: str | None = None
    numeric_forecast_steps: int = 6
    numeric_forecast_mode: str = "auto"
    numeric_model_path: str | None = None
    phase_model_mode: str = "auto"
    phase_model_path: str | None = None
    live_phase_history_path: str | None = None
    live_phase_state_path: str | None = None
    prefer_live_phase: bool = True
    feedback_action: str | None = None
    feedback_label: str = ""
    feedback_note: str = ""
    reviewer: str = "human"


def _summarize_response(result: dict[str, Any]) -> dict[str, Any]:
    phase_payload = result.get("phase_prediction", {})
    numeric_payload = result.get("numeric_forecast", {})
    phase = phase_payload.get("evidence", {}) if isinstance(phase_payload, dict) else {}
    candidates = phase.get("candidate_next_phases", [])
    analogs = phase.get("similar_transition_analogs", [])
    support_values = [int(row.get("support") or 0) for row in candidates if row.get("support") is not None]
    support = max(support_values, default=int(phase.get("support") or 0))
    min_support = int(phase.get("minimum_support", 5) or 5)
    low_support = bool(candidates and phase.get("support") is not None and support < min_support)
    evidence_pack = result.get("evidence_pack")
    if not isinstance(evidence_pack, dict):
        evidence_pack = result.get("evidence")
    if not isinstance(evidence_pack, dict):
        evidence_pack = {}
    intent = result.get("filter", {}).get("intent")
    similar = result.get("similar_windows") or evidence_pack.get("similar_windows", [])
    if intent == "phase_prediction":
        similar = []

    answer = result.get("answer") or (
        phase_payload.get("phase_forecast", {}).get("explanation", "") if isinstance(phase_payload, dict) else ""
    )
    if isinstance(answer, dict):
        answer = answer.get("answer") or answer.get("text") or answer.get("summary") or str(answer)
    if isinstance(answer, list):
        answer = "\n".join(str(item) for item in answer)
    if not isinstance(answer, str):
        answer = str(answer)
    if low_support:
        answer = (
            f"{answer}\n\nLow-support warning: this probability is based on only {support} "
            "historical matching transition(s), so treat it as weak evidence."
        )

    return {
        "answer": answer,
        "intent": intent,
        "confidence": result.get("filter", {}).get("confidence"),
        "mode": result.get("mode"),
        "retrieval_context": result.get("retrieval_context"),
        "explanation_context": result.get("explanation_context"),
        "explanation_mode": result.get("explanation_mode"),
        "forecast_model_source": result.get("forecast_model_source"),
        "model_artifact_path": result.get("model_artifact_path"),
        "model_warning": result.get("model_warning"),
        "agent_graph": result.get("agent_graph"),
        "router": result.get("router") or result.get("agent_graph", {}).get("router"),
        "thread_id": result.get("thread_id"),
        "critic_warnings": result.get("critic_warnings", []),
        "llm_requested": result.get("llm_requested"),
        "llm_available": result.get("llm_available"),
        "window_id": result.get("window_id"),
        "phase_support": support,
        "phase_low_support": low_support,
        "phase_transition_source": phase.get("transition_source") or (
            candidates[0].get("transition_source") if candidates else None
        ),
        "live_token_sequence": phase.get("live_token_sequence"),
        "current_live_state": phase.get("current_live_state"),
        "numeric_forecast": {},
        "top_phase_candidates": candidates[:5],
        "transition_analogs": analogs[:5],
        "similar_windows": similar[:5],
        "memory_rag_debug": result.get("memory_rag_debug"),
        "review_state": result.get("review_state"),
        "human_review_prompt": result.get("human_review_prompt"),
    }


def _resolve_metadata_path(value: str | None = None) -> Path:
    metadata = Path(value).expanduser() if value else _default_metadata_path()
    if not metadata.exists():
        raise HTTPException(status_code=404, detail=f"Metadata file not found: {metadata}")
    return metadata


def _compact_window_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": str(row.get("window_id", "")),
        "window_start": str(row.get("window_start", "")),
        "window_end": str(row.get("window_end", "")),
        "token_id": int(row.get("token_id", -1)) if pd.notna(row.get("token_id")) else None,
        "regime_name": str(row.get("regime_name", "")),
        "wind_speed_mean": round(float(row.get("wind_speed_mean")), 3)
        if pd.notna(row.get("wind_speed_mean"))
        else None,
    }


app = FastAPI(title="Standalone HITL Wind Review UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    live_sync = _sync_live_package_on_load()
    live_semantic_refresh = _refresh_hourly_live_semantic_state()
    metadata = _default_metadata_path()
    return {
        "metadata": str(metadata),
        "metadata_exists": metadata.exists(),
        "live_phase_history_path": str(_default_live_history_path()),
        "live_phase_history_exists": _default_live_history_path().exists(),
        "live_phase_state_path": str(_default_live_state_path()),
        "live_phase_state_exists": _default_live_state_path().exists(),
        "live_raw_path": str(_default_live_raw_path()),
        "live_raw_exists": _default_live_raw_path().exists(),
        "numeric_forecast_steps": 6,
        "numeric_forecast_mode": "disabled",
        "numeric_model_path": "",
        "numeric_model_exists": False,
        "phase_model_mode": "transition",
        "phase_model_path": "",
        "phase_model_exists": False,
        "live_sync": live_sync,
        "live_semantic_refresh": live_semantic_refresh,
        "llm_enabled_by_default": False,
    }


@app.get("/api/windows")
def list_windows(metadata: str | None = None, limit: int = 30, token_id: int | None = None) -> dict[str, Any]:
    metadata_path = _resolve_metadata_path(metadata)
    payload = load_semantic_metadata(metadata_path)
    states_path = Path(payload["output_paths"]["semantic_states"])
    if not states_path.exists():
        raise HTTPException(status_code=404, detail=f"Semantic states file not found: {states_path}")

    frame = pd.read_csv(states_path)
    if frame.empty:
        return {"windows": []}

    if token_id is not None and "token_id" in frame.columns:
        frame = frame[pd.to_numeric(frame["token_id"], errors="coerce") == int(token_id)]

    if "window_start" in frame.columns:
        frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
        frame = frame.sort_values("window_start")

    rows = frame.tail(max(1, min(int(limit), 200))).iloc[::-1]
    return {"windows": [_compact_window_row(row.to_dict()) for _, row in rows.iterrows()]}


@app.post("/api/hitl")
def run_hitl(body: HITLUIRequest) -> dict[str, Any]:
    metadata = _resolve_metadata_path(body.metadata)

    try:
        pipeline = StandaloneHITLPipeline(
            metadata_path=metadata,
            window_id=body.window_id,
            latest=body.latest if not body.window_id else False,
            top_k=body.top_k,
            phase_tokens=body.phase_tokens,
            phase_history_length=body.phase_history_length,
            phase_horizon_steps=body.phase_horizon_steps,
            phase_top_k=body.phase_top_k,
            phase_analog_k=body.phase_analog_k,
            phase_min_support=body.phase_min_support,
            live_phase_history_path=body.live_phase_history_path,
            live_phase_state_path=body.live_phase_state_path,
            live_raw_path=body.live_raw_path,
            numeric_forecast_steps=body.numeric_forecast_steps,
            numeric_forecast_mode=body.numeric_forecast_mode,
            numeric_model_path=body.numeric_model_path,
            phase_model_mode=body.phase_model_mode,
            phase_model_path=body.phase_model_path,
            prefer_live_phase=body.prefer_live_phase,
            enable_rag=body.enable_rag,
            enable_llm=body.enable_llm,
            enable_agent_graph=body.enable_agent_graph,
            confidence_threshold=body.confidence_threshold,
        )
        result = pipeline.process(
            question=body.question,
            feedback_action=body.feedback_action,
            feedback_label=body.feedback_label,
            feedback_note=body.feedback_note,
            reviewer=body.reviewer,
            thread_id=body.thread_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"summary": _summarize_response(result), "raw": result}


def main() -> None:
    import uvicorn

    uvicorn.run("scripts.run_hitl_ui:app", host="127.0.0.1", port=7861, reload=False)


if __name__ == "__main__":
    main()
