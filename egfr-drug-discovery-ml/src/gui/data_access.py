from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.gui.labels import status_label


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def tail_text(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def short_smiles(smiles: str, length: int = 42) -> str:
    return smiles if len(smiles) <= length else f"{smiles[:length]}..."


def prepare_molecule_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["stare_afisare"] = out.get("live_status", pd.Series(["necunoscut"] * len(out))).map(status_label)
    out["transformare_afisare"] = (
        out.get("action_name", pd.Series(["necunoscut"] * len(out))).fillna("necunoscut").astype(str)
    )
    out["smiles_scurt"] = out["smiles"].fillna("").astype(str).map(short_smiles)
    out["risc_proxy"] = pd.to_numeric(out.get("reward_hacking_risk", 0.0), errors="coerce").fillna(0.0).round(3)

    numeric_columns = [
        "rank",
        "live_rank_score",
        "predicted_pIC50",
        "QED",
        "generator_composite_score",
        "estimated_cost_for_10mg_usd",
        "estimated_cost_usd_per_mmol",
        "synthetic_feasibility_score",
        "reward_hacking_risk",
        "max_market_similarity",
        "uncertainty",
        "round",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def summarize_molecules(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "molecule_count": 0,
            "promising_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "best_pic50": 0.0,
            "mean_qed": 0.0,
            "best_qed": 0.0,
        }

    counts = df.get("live_status", pd.Series(dtype=str)).fillna("necunoscut").value_counts().to_dict()
    return {
        "molecule_count": len(df),
        "promising_count": int(counts.get("promovata", 0)),
        "review_count": int(counts.get("revizie", 0)),
        "rejected_count": int(counts.get("respinsa", 0)),
        "best_pic50": safe_float(df.get("predicted_pIC50", pd.Series(dtype=float)).max(), 0.0),
        "mean_qed": safe_float(df.get("QED", pd.Series(dtype=float)).mean(), 0.0),
        "best_qed": safe_float(df.get("QED", pd.Series(dtype=float)).max(), 0.0),
    }
