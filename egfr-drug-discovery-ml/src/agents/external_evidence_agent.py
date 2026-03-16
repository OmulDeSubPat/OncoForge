from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.multi_agent import resolve_priority_score_column


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def add_external_evidence_agent_ranking(
    df: pd.DataFrame,
    base_score_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    base_column = base_score_col or resolve_priority_score_column(out)

    cross_db_consensus = _series(out, "cross_database_consensus_score", 0.0).clip(lower=0.0, upper=1.0)
    pubchem_support = _series(out, "pubchem_support_score", 0.0).clip(lower=0.0, upper=1.0)
    pubchem_match_evidence = _series(out, "pubchem_best_match_evidence", 0.0).clip(lower=0.0, upper=1.0)
    iuphar_support = _series(out, "iuphar_support_score", 0.0).clip(lower=0.0, upper=1.0)
    bindingdb_support = _series(out, "bindingdb_support_score", 0.0).clip(lower=0.0, upper=1.0)
    papyrus_support = _series(out, "papyrus_support_score", 0.0).clip(lower=0.0, upper=1.0)
    excape_support = _series(out, "excape_support_score", 0.0).clip(lower=0.0, upper=1.0)
    source_support = _series(out, "source_support_score", 0.0).clip(lower=0.0, upper=1.0)
    active_similarity = _series(out, "max_active_similarity", 0.0).clip(lower=0.0, upper=1.0)
    independent_support = (_series(out, "cross_database_independent_support_count", 0.0) / 4.0).clip(lower=0.0, upper=1.0)
    secondary_support = (_series(out, "cross_database_secondary_support_count", 0.0) / 2.0).clip(lower=0.0, upper=1.0)
    external_support = (_series(out, "cross_database_external_support_count", 0.0) / 5.0).clip(lower=0.0, upper=1.0)
    risk_guardrail = (1.0 - _series(out, "reward_hacking_risk", 0.5)).clip(lower=0.0, upper=1.0)

    out["external_evidence_support"] = (
        0.22 * cross_db_consensus
        + 0.16 * pubchem_support
        + 0.08 * pubchem_match_evidence
        + 0.10 * iuphar_support
        + 0.09 * bindingdb_support
        + 0.08 * papyrus_support
        + 0.06 * excape_support
        + 0.09 * independent_support
        + 0.04 * secondary_support
        + 0.04 * external_support
        + 0.05 * source_support
        + 0.05 * active_similarity
    ).clip(lower=0.0, upper=1.0)
    out["external_evidence_guardrail"] = (
        0.65 * risk_guardrail
        + 0.25 * independent_support
        + 0.10 * secondary_support
    ).clip(lower=0.0, upper=1.0)
    out["external_evidence_status"] = np.select(
        [
            ((_series(out, "veto", 0.0) >= 1.0))
            | (
                (out.get("cross_database_status", pd.Series("weak", index=out.index)) == "weak")
                & (pubchem_support < 0.30)
                & (bindingdb_support < 0.30)
            ),
            (out["external_evidence_support"] < 0.45)
            | (out["external_evidence_guardrail"] < 0.45),
        ],
        ["fail", "review"],
        default="pass",
    )
    out["external_evidence_priority"] = (
        _series(out, base_column, 0.0)
        + 0.95 * out["external_evidence_support"]
        + 0.30 * independent_support
        + 0.12 * secondary_support
        + 0.15 * external_support
    )
    out["external_evidence_percentile"] = out["external_evidence_support"].rank(method="average", pct=True, ascending=True)
    out["external_evidence_gap"] = out["external_evidence_support"] - out["external_evidence_guardrail"]
    return out
