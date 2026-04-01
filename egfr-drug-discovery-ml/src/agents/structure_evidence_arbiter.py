from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.multi_agent import resolve_priority_score_column
from src.utils.pareto_selection import add_pareto_front_columns


def _series(df: pd.DataFrame, column: str, default: float | pd.Series = 0.0) -> pd.Series:
    if isinstance(default, pd.Series):
        default_series = pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
    else:
        default_series = pd.Series(float(default), index=df.index, dtype=float)
    if column not in df.columns:
        return default_series
    return pd.to_numeric(df[column], errors="coerce").fillna(default_series)


def add_structure_evidence_arbiter(
    df: pd.DataFrame,
    *,
    base_score_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    base_column = base_score_col or resolve_priority_score_column(out)

    structure_support = (
        0.38 * _series(out, "docking_rescore", 0.0).clip(lower=0.0, upper=1.0)
        + 0.34 * _series(out, "interaction_support_score", 0.0).clip(lower=0.0, upper=1.0)
        + 0.28 * _series(out, "structural_guidance_score", _series(out, "structure_agent_support", 0.0)).clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    evidence_support = (
        0.38 * _series(out, "external_evidence_support", 0.0).clip(lower=0.0, upper=1.0)
        + 0.34 * _series(out, "cross_database_consensus_score", 0.0).clip(lower=0.0, upper=1.0)
        + 0.28 * _series(out, "evidence_arbiter_support", 0.0).clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    readiness_support = _series(out, "experimental_readiness_score", 0.0).clip(lower=0.0, upper=1.0)
    feasibility_support = _series(out, "feasibility_score", 0.0).clip(lower=0.0, upper=1.0)
    novelty_support = _series(out, "novelty_score", _series(out, "market_novelty_score", 0.0)).clip(lower=0.0, upper=1.0)
    adaptive_prior = _series(out, "adaptive_action_prior", 0.50).clip(lower=0.0, upper=1.0)
    generator_support = _series(out, "generator_priority_score", 0.0).clip(lower=0.0, upper=1.0)
    hacking_guard = (1.0 - _series(out, "reward_hacking_risk", 0.5)).clip(lower=0.0, upper=1.0)
    uncertainty = _series(out, "uncertainty", 0.15)
    uncertainty_guard = (
        1.0
        - (
            uncertainty
            / max(0.10, float(uncertainty.quantile(0.90)) if not uncertainty.empty else 0.15)
        )
    ).clip(lower=0.0, upper=1.0)
    audit_guard = (
        (out.get("audit_status", pd.Series("review", index=out.index)) == "pass").astype(float)
        - 0.50 * (out.get("audit_status", pd.Series("review", index=out.index)) == "review").astype(float)
    ).clip(lower=0.0, upper=1.0)
    feasibility_gate = _series(out, "feasibility_hard_gate_pass", 1.0).clip(lower=0.0, upper=1.0)

    out["structure_evidence_support"] = (
        0.34 * structure_support
        + 0.28 * evidence_support
        + 0.16 * readiness_support
        + 0.10 * feasibility_support
        + 0.06 * novelty_support
        + 0.06 * adaptive_prior
    ).clip(lower=0.0, upper=1.0)
    out["structure_evidence_guardrail"] = (
        0.38 * hacking_guard
        + 0.22 * uncertainty_guard
        + 0.18 * feasibility_support
        + 0.12 * feasibility_gate
        + 0.10 * audit_guard
    ).clip(lower=0.0, upper=1.0)
    out["structure_evidence_gap"] = out["structure_evidence_support"] - out["structure_evidence_guardrail"]

    out = add_pareto_front_columns(
        out,
        maximize=[
            base_column,
            "structure_evidence_support",
            "experimental_readiness_score",
            "feasibility_score",
            "novelty_score" if "novelty_score" in out.columns else "market_novelty_score",
        ],
        minimize=["reward_hacking_risk", "uncertainty", "max_market_similarity"],
        prefix="structure_evidence_pareto",
    )

    out["structure_evidence_status"] = np.select(
        [
            (_series(out, "veto", 0.0) >= 1.0)
            | (out.get("audit_status", pd.Series("review", index=out.index)) == "fail")
            | (out.get("experimental_readiness_status", pd.Series("supporting", index=out.index)) == "hold")
            | (out["structure_evidence_guardrail"] < 0.38)
            | (_series(out, "feasibility_status", pd.Series("review", index=out.index)) == "fail"),
            (out["structure_evidence_support"] < 0.52)
            | (out["structure_evidence_guardrail"] < 0.48)
            | (out["structure_evidence_pareto_front_rank"] > 2),
        ],
        ["fail", "review"],
        default="pass",
    )
    out["structure_evidence_priority"] = (
        _series(out, base_column, 0.0)
        + 1.10 * out["structure_evidence_support"]
        + 0.20 * out["structure_evidence_pareto_priority_bonus"]
        + 0.18 * adaptive_prior
        + 0.16 * generator_support
        + 0.12 * novelty_support
        + 0.10 * readiness_support
    )
    out["structure_evidence_state_priority"] = out["structure_evidence_status"].map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    out = out.sort_values(
        [
            "structure_evidence_state_priority",
            "structure_evidence_pareto_front_rank",
            "structure_evidence_priority",
            "predicted_pIC50" if "predicted_pIC50" in out.columns else "structure_evidence_support",
        ],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    out["structure_evidence_rank"] = np.arange(1, len(out) + 1)
    return out
