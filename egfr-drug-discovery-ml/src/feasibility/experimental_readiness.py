from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.agents.multi_agent import add_structure_agent_ranking, resolve_priority_score_column
from src.config import PROJECT_ROOT


def _clip01_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def load_market_benchmark(path: Path | None = None) -> pd.DataFrame | None:
    candidates = [
        path,
        PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv",
        PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return pd.read_csv(candidate, low_memory=False)
    return None


def add_experimental_readiness(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    sort_output: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = add_structure_agent_ranking(df, base_score_col=resolve_priority_score_column(df))
    if "cross_database_consensus_score" in out.columns:
        out = add_external_evidence_agent_ranking(out)
    market = market_df if market_df is not None else load_market_benchmark()

    feasibility_score = _clip01_series(_series(out, "feasibility_score", 0.0))
    docking_support = _clip01_series(_series(out, "docking_rescore", 0.0))
    interaction_support = _clip01_series(_series(out, "interaction_support_score", 0.0))
    key_residue_support = _clip01_series(_series(out, "interaction_key_residue_support", 0.0))
    if "interaction_key_residue_support" not in out.columns:
        key_residue_support = (_series(out, "interaction_key_residue_count", 0.0) / 4.0).clip(lower=0.0, upper=1.0)
    active_support = _clip01_series(_series(out, "max_active_similarity", 0.0))
    source_support = _clip01_series(_series(out, "source_support_score", 0.0))
    cross_db_consensus = _clip01_series(_series(out, "cross_database_consensus_score", 0.0))
    cross_db_independent_support = (_series(out, "cross_database_independent_support_count", 0.0) / 3.0).clip(lower=0.0, upper=1.0)
    external_evidence_support = _clip01_series(_series(out, "external_evidence_support", 0.0))
    traceability = _clip01_series(_series(out, "traceability_score", 0.0))
    qed_support = _clip01_series(_series(out, "QED", 0.0))
    synthetic_support = _clip01_series(_series(out, "synthetic_ease_score", 0.0))
    risk_support = (1.0 - _clip01_series(_series(out, "reward_hacking_risk", 0.5))).clip(lower=0.0, upper=1.0)

    uncertainty = _series(out, "uncertainty", 0.20)
    uncertainty_scale = max(0.10, float(uncertainty.quantile(0.90))) if not uncertainty.empty else 0.20
    low_uncertainty_support = (1.0 - (uncertainty / uncertainty_scale)).clip(lower=0.0, upper=1.0)

    market_docking_ref = None
    market_interaction_ref = None
    if market is not None and not market.empty:
        if "docking_rescore" in market.columns and not market["docking_rescore"].dropna().empty:
            market_docking_ref = float(market["docking_rescore"].quantile(0.25))
        if "interaction_support_score" in market.columns and not market["interaction_support_score"].dropna().empty:
            market_interaction_ref = float(market["interaction_support_score"].quantile(0.25))

    if market_docking_ref is None:
        market_docking_alignment = docking_support.copy()
    else:
        market_docking_alignment = ((docking_support - (market_docking_ref - 0.05)) / 0.40).clip(lower=0.0, upper=1.0)

    if market_interaction_ref is None:
        market_interaction_alignment = interaction_support.copy()
    else:
        market_interaction_alignment = ((interaction_support - (market_interaction_ref - 0.05)) / 0.35).clip(lower=0.0, upper=1.0)

    market_alignment = (0.55 * market_docking_alignment + 0.45 * market_interaction_alignment).clip(lower=0.0, upper=1.0)

    out["market_docking_alignment"] = market_docking_alignment
    out["market_interaction_alignment"] = market_interaction_alignment
    out["market_alignment_support"] = market_alignment

    out["experimental_readiness_score"] = (
        0.23 * feasibility_score
        + 0.14 * docking_support
        + 0.12 * interaction_support
        + 0.08 * cross_db_consensus
        + 0.08 * external_evidence_support
        + 0.04 * cross_db_independent_support
        + 0.08 * market_alignment
        + 0.10 * active_support
        + 0.06 * source_support
        + 0.06 * traceability
        + 0.07 * qed_support
        + 0.03 * synthetic_support
        + 0.03 * risk_support
        + 0.04 * low_uncertainty_support
    ).clip(lower=0.0, upper=1.0)

    evidence = []
    for idx, row in out.iterrows():
        row_evidence: list[str] = []
        if float(feasibility_score.iloc[idx]) >= 0.70:
            row_evidence.append("feasibility_pass")
        if float(market_docking_alignment.iloc[idx]) >= 0.55:
            row_evidence.append("market_level_docking")
        if float(market_interaction_alignment.iloc[idx]) >= 0.55:
            row_evidence.append("market_level_interactions")
        if float(active_support.iloc[idx]) >= 0.55:
            row_evidence.append("near_known_active")
        if float(cross_db_consensus.iloc[idx]) >= 0.55:
            row_evidence.append("cross_database_consensus")
        if float(external_evidence_support.iloc[idx]) >= 0.55:
            row_evidence.append("external_evidence_support")
        if float(cross_db_independent_support.iloc[idx]) >= 0.34:
            row_evidence.append("independent_database_support")
        if float(source_support.iloc[idx]) >= 0.35:
            row_evidence.append("multi_source_neighbor_support")
        if float(traceability.iloc[idx]) >= 0.50:
            row_evidence.append("traceable_medchem_route")
        if float(qed_support.iloc[idx]) >= 0.45 and float(synthetic_support.iloc[idx]) >= 0.40:
            row_evidence.append("drug_like_chemistry")
        if float(risk_support.iloc[idx]) >= 0.65:
            row_evidence.append("low_proxy_risk")
        evidence.append(";".join(row_evidence) if row_evidence else None)

    out["experimental_readiness_evidence"] = evidence
    out["experimental_readiness_evidence_count"] = out["experimental_readiness_evidence"].fillna("").apply(
        lambda value: len([item for item in value.split(";") if item])
    )
    out["experimental_readiness_status"] = np.select(
        [
            ((_series(out, "veto", 0.0) >= 1.0) | (out.get("audit_status", pd.Series("review", index=out.index)) == "fail")),
            (
                (out["experimental_readiness_score"] >= 0.72)
                & (out.get("feasibility_status", pd.Series("review", index=out.index)) == "pass")
                & (out.get("audit_status", pd.Series("review", index=out.index)) == "pass")
                & (docking_support >= 0.50)
                & (interaction_support >= 0.42)
                & (cross_db_consensus >= 0.45)
            ),
            (
                (out["experimental_readiness_score"] >= 0.58)
                & (out.get("audit_status", pd.Series("review", index=out.index)) != "fail")
            ),
        ],
        ["hold", "ready", "supporting"],
        default="hold",
    )
    out["experimental_track"] = np.select(
        [
            (out["experimental_readiness_status"] == "ready") & (_series(out, "novelty_score", 0.0) >= 0.55),
            (out["experimental_readiness_status"] == "ready"),
            (out["experimental_readiness_status"] == "supporting"),
        ],
        ["prospective_explore", "benchmark_ready", "supporting_evidence"],
        default="hold",
    )
    out["experimental_readiness_priority"] = (
        _series(out, resolve_priority_score_column(out), 0.0)
        + 1.10 * out["experimental_readiness_score"]
        + 0.35 * cross_db_consensus
        + 0.20 * external_evidence_support
        + 0.35 * market_alignment
        + 0.20 * source_support
    )

    if sort_output:
        out["experimental_priority"] = out["experimental_readiness_status"].map(
            {"ready": 0, "supporting": 1, "hold": 2}
        ).fillna(1).astype(int)
        out = out.sort_values(
            [
                "experimental_priority",
                "experimental_readiness_priority",
                "structure_augmented_score",
                "predicted_pIC50" if "predicted_pIC50" in out.columns else "experimental_readiness_score",
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        out["experimental_readiness_rank"] = np.arange(1, len(out) + 1)

    return out
