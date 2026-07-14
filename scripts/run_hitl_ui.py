from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.request import urlopen

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
DEFAULT_GITHUB_RAW_BASE = os.getenv("WIND_LIVE_DATA_BASE_URL", "").rstrip("/")
DEFAULT_GITHUB_LIVE_FILES = {
    "history": (
        "data/live/semantic_states/KBOS_live_semantic_history.csv",
        DEFAULT_LOCAL_LIVE_HISTORY,
    ),
    "state": (
        "data/live/semantic_states/KBOS_live_semantic_state.json",
        DEFAULT_LOCAL_LIVE_STATE,
    ),
    "raw": (
        "data/live/aviationweather_metar/KBOS/metar_live.csv",
        DEFAULT_LOCAL_LIVE_RAW,
    ),
}
STARTUP_LIVE_SYNC: dict[str, Any] | None = None


def _default_live_history_path() -> Path:
    return DEFAULT_LOCAL_LIVE_HISTORY


def _default_live_state_path() -> Path:
    return DEFAULT_LOCAL_LIVE_STATE


def _default_live_raw_path() -> Path:
    return DEFAULT_LOCAL_LIVE_RAW


def _default_metadata_path() -> Path:
    return DEFAULT_5MIN_METADATA if DEFAULT_5MIN_METADATA.exists() else DEFAULT_HOURLY_METADATA


def _download_github_live_files(timeout: int = 20) -> dict[str, Any]:
    """Refresh live HITL files from GitHub raw content, with local-cache fallback.

    This is intentionally best-effort: failed downloads do not block the UI. Any
    successfully downloaded files overwrite the local live-cache files.
    """

    if not DEFAULT_GITHUB_RAW_BASE:
        return {
            "attempted": False,
            "ok": False,
            "status": "remote_live_not_configured",
            "message": "No WIND_LIVE_DATA_BASE_URL was configured.",
            "files": [],
            "failures": [],
        }

    results: list[dict[str, Any]] = []
    ok_count = 0
    for name, (relative_path, destination) in DEFAULT_GITHUB_LIVE_FILES.items():
        url = f"{DEFAULT_GITHUB_RAW_BASE}/{relative_path}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(url, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            destination.write_bytes(data)
            ok_count += 1
            results.append(
                {
                    "name": name,
                    "ok": True,
                    "path": str(destination),
                    "bytes": len(data),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "path": str(destination),
                    "error": str(exc),
                    "fallback_exists": destination.exists(),
                }
            )

    failed = [row for row in results if not row["ok"]]
    if ok_count == len(DEFAULT_GITHUB_LIVE_FILES):
        status = "remote_live_synced"
        message = "Latest live HITL files were downloaded from the configured remote source."
    elif ok_count > 0:
        status = "remote_live_partial"
        message = f"Downloaded {ok_count}/{len(DEFAULT_GITHUB_LIVE_FILES)} live files; local cache used for missing files."
    else:
        status = "remote_live_fallback"
        message = "Could not download configured live files; using the local/API fallback."

    return {
        "attempted": True,
        "ok": ok_count > 0 or all(destination.exists() for _, destination in DEFAULT_GITHUB_LIVE_FILES.values()),
        "status": status,
        "message": message,
        "files": results,
        "failures": failed,
    }


def _run_live_command(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": None, "message": str(exc)}

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if len(output) > 800:
        output = output[-800:]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "message": output,
    }


def _fetch_live_metar_from_aviationweather() -> dict[str, Any]:
    output_dir = PROJECT_ROOT / "data" / "live" / "aviationweather_metar"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "fetch_aviationweather_metar.py"),
        "--station",
        "KBOS",
        "--hours",
        "18",
        "--output_dir",
        str(output_dir),
        "--timeout",
        "30",
    ]
    result = _run_live_command(command, timeout=45)
    return {
        "attempted": True,
        "ok": bool(result["ok"]),
        "status": "metar_fetched" if result["ok"] else "metar_fetch_failed",
        "message": result["message"] or ("Live METAR rows fetched." if result["ok"] else "METAR fetch failed."),
        "output_csv": str(DEFAULT_LOCAL_LIVE_RAW),
    }


def _build_live_semantic_state_from_raw() -> dict[str, Any]:
    metadata = _default_metadata_path()
    if not metadata.exists():
        return {
            "attempted": False,
            "ok": False,
            "status": "missing_metadata",
            "message": f"Semantic metadata not found: {metadata}",
        }

    live_raw = DEFAULT_LOCAL_LIVE_RAW if DEFAULT_LOCAL_LIVE_RAW.exists() else _default_live_raw_path()
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
        str(metadata),
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
    result = _run_live_command(command, timeout=75)
    return {
        "attempted": True,
        "ok": bool(result["ok"]),
        "status": "semantic_state_rebuilt" if result["ok"] else "semantic_state_rebuild_failed",
        "message": result["message"] or (
            "Live semantic state rebuilt from the latest METAR rows."
            if result["ok"]
            else "Live semantic state rebuild failed."
        ),
        "metadata": str(metadata),
        "live_raw": str(live_raw),
    }


def _refresh_live_semantic_state_from_aviationweather() -> dict[str, Any]:
    """Fetch METAR rows and rebuild live semantic state if GitHub live state is unavailable."""

    fetch = _fetch_live_metar_from_aviationweather()
    if not fetch["ok"]:
        return {
            "attempted": True,
            "ok": False,
            "status": "aviationweather_failed",
            "message": f"Could not fetch fresh METAR rows: {fetch['message']}",
            "fetch": fetch,
        }

    build = _build_live_semantic_state_from_raw()
    return {
        "attempted": True,
        "ok": bool(build["ok"]),
        "status": "live_api_refreshed" if build["ok"] else "live_api_build_failed",
        "message": (
            "Fresh AviationWeather METAR rows were fetched and the live semantic state was rebuilt."
            if build["ok"]
            else f"METAR fetch succeeded, but semantic rebuild failed: {build['message']}"
        ),
        "fetch": fetch,
        "build": build,
    }


def _refresh_live_state_with_fallback() -> dict[str, Any]:
    github_live = _download_github_live_files()
    if github_live["ok"]:
        return {
            "attempted": True,
            "ok": True,
            "status": "remote_live_state",
            "message": "Live semantic state was refreshed from the configured remote source.",
            "source": "configured_remote",
            "fresh": None,
            "github": github_live,
            "fallback": None,
        }

    fallback = _refresh_live_semantic_state_from_aviationweather()
    return {
        "attempted": True,
        "ok": bool(fallback["ok"]),
        "status": "aviationweather_fallback" if fallback["ok"] else "live_refresh_failed",
        "message": (
            f"Remote live refresh was unavailable ({github_live['status']}); rebuilt from AviationWeather instead."
            if fallback["ok"]
            else f"Remote live refresh failed ({github_live['status']}) and AviationWeather fallback also failed."
        ),
        "source": "aviationweather" if fallback["ok"] else "none",
        "fresh": fallback if fallback["ok"] else None,
        "github": github_live,
        "fallback": fallback,
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
    numeric_forecast_mode: str = "disabled"
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

    review_state = result.get("review_state") if isinstance(result.get("review_state"), dict) else {}
    current_live_state = phase.get("current_live_state") if isinstance(phase.get("current_live_state"), dict) else {}
    display_window_id = (
        review_state.get("window_id")
        or current_live_state.get("window_id")
        or result.get("window_id")
    )
    route = result.get("agent_graph", {}).get("route") if isinstance(result.get("agent_graph"), dict) else None
    if route == "feedback" and display_window_id and not str(display_window_id).startswith("live_"):
        # Feedback does not rerun a live forecast; avoid showing the historical
        # context window as if it were the reviewed live input window.
        display_window_id = None

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
        "window_id": display_window_id,
        "phase_support": support,
        "phase_low_support": low_support,
        "phase_transition_source": phase.get("transition_source") or (
            candidates[0].get("transition_source") if candidates else None
        ),
        "live_token_sequence": phase.get("live_token_sequence"),
        "current_live_state": current_live_state or phase.get("current_live_state"),
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global STARTUP_LIVE_SYNC
    STARTUP_LIVE_SYNC = _refresh_live_state_with_fallback()
    print(f"[HITL live sync] {STARTUP_LIVE_SYNC['status']}: {STARTUP_LIVE_SYNC['message']}", flush=True)
    yield


app = FastAPI(title="Standalone HITL Wind Review UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    live_sync = STARTUP_LIVE_SYNC or _refresh_live_state_with_fallback()
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
        "live_semantic_refresh": live_sync.get("fresh"),
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

    parser = argparse.ArgumentParser(description="Run the controlled HITL review interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run("scripts.run_hitl_ui:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
