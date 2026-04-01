from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT
from src.economics.cost_model import LITERATURE_SOURCE_URLS
from src.gui.live_generation_worker import MODE_LABELS


GUI_ROOT = PROJECT_ROOT / "reports" / "gui_live"
DEFAULT_SESSION_NAME = "sesiune_curenta"
DEFAULT_SESSION_DIR = GUI_ROOT / DEFAULT_SESSION_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def is_worker_pid_active(pid: int) -> bool:
    if pid <= 0:
        return False

    if sys.platform.startswith("win"):
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (completed.stdout or "").strip()
        if not output or output.startswith("INFO:"):
            return False
        return str(pid) in output

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_manual_status(session_dir: Path, status: str, **extra: object) -> None:
    payload: dict[str, object] = {
        "status": status,
        "updated_at": _now_iso(),
        "session_dir": str(session_dir),
        "source_urls": LITERATURE_SOURCE_URLS,
    }
    payload.update(extra)
    _write_json(session_dir / "status.json", payload)


def resolve_session_dir(session_name: str | None = None) -> Path:
    name = (session_name or DEFAULT_SESSION_NAME).strip() or DEFAULT_SESSION_NAME
    safe_name = name.replace("\\", "/").split("/")[-1]
    session_dir = (GUI_ROOT / safe_name).resolve()
    gui_root = GUI_ROOT.resolve()
    if gui_root not in session_dir.parents and session_dir != gui_root:
        raise ValueError("Sesiunea trebuie sa fie in reports/gui_live.")
    return session_dir


def launch_worker(
    *,
    session_dir: Path,
    mode: str,
    seed_count: int,
    rounds: int,
    variants_per_seed: int,
    beam_width: int,
) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.gui.live_generation_worker",
        "--session-dir",
        str(session_dir),
        "--mode",
        str(mode),
        "--seed-count",
        str(seed_count),
        "--rounds",
        str(rounds),
        "--variants-per-seed",
        str(variants_per_seed),
        "--beam-width",
        str(beam_width),
        "--no-reset-session",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return int(proc.pid)


def write_launch_status(
    *,
    session_dir: Path,
    worker_pid: int,
    mode: str,
    seed_count: int,
    rounds: int,
    variants_per_seed: int,
    beam_width: int,
) -> None:
    write_manual_status(
        session_dir,
        "pornit",
        pid=int(worker_pid),
        mesaj="Worker-ul a fost lansat si pregateste prima runda",
        mod=mode,
        mod_label=MODE_LABELS.get(mode, mode),
        current_round=0,
        total_rounds=int(rounds if mode == "iterativ" else 1),
        current_seed=0,
        total_seeds=int(seed_count),
        seed_count=int(seed_count),
        variants_per_seed=int(variants_per_seed),
        beam_width=int(beam_width),
        molecule_count=0,
        attempted_candidates=0,
        generated_candidates=0,
    )


def terminate_worker(pid: int) -> None:
    if pid <= 0:
        return
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=creationflags,
        )
        return

    import os
    import signal

    os.kill(pid, signal.SIGTERM)


def reset_session_dir(session_dir: Path) -> None:
    resolved = session_dir.resolve()
    gui_root = GUI_ROOT.resolve()
    if gui_root not in resolved.parents and resolved != gui_root:
        raise ValueError("Directorul sesiunii trebuie sa fie in reports/gui_live.")
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
