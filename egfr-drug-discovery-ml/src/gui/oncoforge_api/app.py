from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_BOOTSTRAP_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.gui.oncoforge_api.runtime import launch_worker, reset_session_dir, resolve_session_dir, terminate_worker, write_launch_status, write_manual_status
from src.gui.oncoforge_api.service import build_control_state, build_dashboard_payload, build_molecule_payload, build_sources_payload


class StartSessionPayload(BaseModel):
    session_name: str = Field(default="sesiune_curenta")
    mode: str = Field(default="ghidat_ai")
    seed_count: int = Field(default=3, ge=1, le=20)
    rounds: int = Field(default=3, ge=1, le=12)
    variants_per_seed: int = Field(default=12, ge=1, le=64)
    beam_width: int = Field(default=8, ge=1, le=32)
    replace_existing: bool = True


class SessionSelectorPayload(BaseModel):
    session_name: str = Field(default="sesiune_curenta")


app = FastAPI(
    title="OncoSynth API",
    version="0.1.0",
    description="API local pentru dashboard-ul OncoSynth de generare si analiza moleculara.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "oncosynth-api"}


@app.get("/api/dashboard")
def dashboard(
    session_name: str = Query(default="sesiune_curenta"),
    limit: int = Query(default=100, ge=10, le=250),
    search: str = Query(default=""),
    status_filter: str = Query(default="all"),
    rank: int | None = Query(default=None),
    smiles: str | None = Query(default=None),
) -> dict[str, Any]:
    return build_dashboard_payload(
        session_name=session_name,
        limit=limit,
        search=search,
        status_filter=status_filter,
        rank=rank,
        smiles=smiles,
    )


@app.get("/api/molecule")
def molecule_detail(
    session_name: str = Query(default="sesiune_curenta"),
    smiles: str = Query(..., min_length=1),
) -> dict[str, Any]:
    return build_molecule_payload(session_name, smiles)


@app.get("/api/control")
def control_state(session_name: str = Query(default="sesiune_curenta")) -> dict[str, Any]:
    return build_control_state(session_name)


@app.get("/api/sources")
def sources(session_name: str = Query(default="sesiune_curenta")) -> dict[str, Any]:
    return build_sources_payload(session_name)


@app.post("/api/control/start")
def start_session(payload: StartSessionPayload) -> dict[str, Any]:
    control = build_control_state(payload.session_name)
    if control["running"]:
        raise HTTPException(status_code=409, detail="Exista deja o sesiune in rulare.")

    session_dir = resolve_session_dir(payload.session_name)
    if payload.replace_existing:
        reset_session_dir(session_dir)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)

    worker_pid = launch_worker(
        session_dir=session_dir,
        mode=payload.mode,
        seed_count=payload.seed_count,
        rounds=payload.rounds,
        variants_per_seed=payload.variants_per_seed,
        beam_width=payload.beam_width,
    )
    write_launch_status(
        session_dir=session_dir,
        worker_pid=worker_pid,
        mode=payload.mode,
        seed_count=payload.seed_count,
        rounds=payload.rounds,
        variants_per_seed=payload.variants_per_seed,
        beam_width=payload.beam_width,
    )
    return {"ok": True, "message": "Sesiunea OncoSynth a fost pornita.", "sessionName": payload.session_name}


@app.post("/api/control/stop")
def stop_session(payload: SessionSelectorPayload) -> dict[str, Any]:
    control = build_control_state(payload.session_name)
    if not control["running"]:
        return {"ok": True, "message": "Nu exista nicio sesiune activa."}
    terminate_worker(int(control["pid"]))
    session_dir = resolve_session_dir(payload.session_name)
    write_manual_status(
        session_dir,
        "oprit",
        pid=int(control["pid"]),
        mesaj="Worker-ul a fost oprit manual din interfata",
        mod=str(control.get("mode", "ghidat_ai")),
        mod_label=str(control.get("modeLabel", "")),
    )
    return {"ok": True, "message": "Procesul worker a fost oprit.", "pid": control["pid"]}


@app.post("/api/control/reset")
def reset_session(payload: SessionSelectorPayload) -> dict[str, Any]:
    control = build_control_state(payload.session_name)
    if control["running"]:
        raise HTTPException(status_code=409, detail="Opreste mai intai sesiunea in rulare.")
    session_dir = resolve_session_dir(payload.session_name)
    reset_session_dir(session_dir)
    return {"ok": True, "message": "Sesiunea a fost resetata.", "sessionName": payload.session_name}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.gui.oncoforge_api.app:app", host="127.0.0.1", port=8000, reload=False)
