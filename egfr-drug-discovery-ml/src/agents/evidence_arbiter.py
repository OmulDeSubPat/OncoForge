from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.multi_agent import resolve_priority_score_column


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if isinstance(default, pd.Series):
        default_series = pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
    else:
        default_series = pd.Series(float(default), index=df.index, dtype=float)
    if column not in df.columns:
        return default_series
    return pd.to_numeric(df[column], errors="coerce").fillna(default_series)


def add_evidence_arbiter_ranking(
    df: pd.DataFrame,
    base_score_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    base_column = base_score_col or resolve_priority_score_column(out)

    structure_support = _series(out, "structure_agent_support", _series(out, "docking_rescore", 0.0)).clip(lower=0.0, upper=1.0)
    interaction_support = _series(out, "interaction_support_score", 0.0).clip(lower=0.0, upper=1.0)
    external_support = _series(out, "external_evidence_support", 0.0).clip(lower=0.0, upper=1.0)
    readiness_support = _series(out, "experimental_readiness_score", 0.0).clip(lower=0.0, upper=1.0)
    market_alignment = _series(out, "market_alignment_support", 0.0).clip(lower=0.0, upper=1.0)
    crossdb_support = _series(out, "cross_database_consensus_score", 0.0).clip(lower=0.0, upper=1.0)
    feasibility_support = _series(out, "feasibility_score", 0.0).clip(lower=0.0, upper=1.0)
    guardrail = (
        0.45 * (1.0 - _series(out, "reward_hacking_risk", 0.5)).clip(lower=0.0, upper=1.0)
        + 0.25 * (1.0 - _series(out, "uncertainty", 0.2) / max(0.10, float(_series(out, "uncertainty", 0.2).quantile(0.90) or 0.2))).clip(lower=0.0, upper=1.0)
        + 0.30 * feasibility_support
    ).clip(lower=0.0, upper=1.0)

    out["evidence_arbiter_support"] = (
        0.22 * structure_support
        + 0.16 * interaction_support
        + 0.18 * external_support
        + 0.18 * readiness_support
        + 0.10 * market_alignment
        + 0.10 * crossdb_support
        + 0.06 * feasibility_support
    ).clip(lower=0.0, upper=1.0)
    out["evidence_arbiter_guardrail"] = guardrail
    out["evidence_arbiter_gap"] = out["evidence_arbiter_support"] - out["evidence_arbiter_guardrail"]
    out["evidence_arbiter_status"] = np.select(
        [
            (_series(out, "veto", 0.0) >= 1.0)
            | (out.get("audit_status", pd.Series("review", index=out.index)) == "fail")
            | (out.get("external_evidence_status", pd.Series("review", index=out.index)) == "fail"),
            (out["evidence_arbiter_support"] < 0.46)
            | (out["evidence_arbiter_guardrail"] < 0.48)
            | (out.get("experimental_readiness_status", pd.Series("supporting", index=out.index)) == "hold"),
        ],
        ["fail", "review"],
        default="pass",
    )
    out["evidence_arbiter_priority"] = (
        _series(out, base_column, 0.0)
        + 1.05 * out["evidence_arbiter_support"]
        + 0.25 * readiness_support
        + 0.20 * external_support
        + 0.15 * structure_support
    )
    out["evidence_arbiter_percentile"] = out["evidence_arbiter_support"].rank(method="average", pct=True, ascending=True)
    out["evidence_arbiter_state_priority"] = out["evidence_arbiter_status"].map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    out = out.sort_values(
        [
            "evidence_arbiter_state_priority",
            "evidence_arbiter_priority",
            "predicted_pIC50" if "predicted_pIC50" in out.columns else "evidence_arbiter_support",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    out["evidence_arbiter_rank"] = np.arange(1, len(out) + 1)
    return out
