from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


GENERATION_MEMORY_ARTIFACTS = [
    PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_crossdb.csv",
    PROJECT_ROOT / "reports" / "ai_guided_analogs_structural_crossdb.csv",
    PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_crossdb.csv",
]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _series(df: pd.DataFrame, column: str, default: float | pd.Series = 0.0) -> pd.Series:
    if isinstance(default, pd.Series):
        default_series = pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
    else:
        default_series = pd.Series(float(default), index=df.index, dtype=float)
    if column not in df.columns:
        return default_series
    return pd.to_numeric(df[column], errors="coerce").fillna(default_series)


@dataclass(frozen=True)
class TransformationMemory:
    action_name_priors: dict[str, float]
    category_priors: dict[str, float]
    reaction_family_priors: dict[str, float]
    rule_source_priors: dict[str, float]
    default_prior: float = 0.50

    def lookup(
        self,
        *,
        action_name: str | None = None,
        category: str | None = None,
        reaction_family: str | None = None,
        rule_source: str | None = None,
    ) -> float:
        votes: list[float] = []
        if action_name:
            value = self.action_name_priors.get(action_name)
            if value is not None:
                votes.append(value)
        if category:
            value = self.category_priors.get(category)
            if value is not None:
                votes.append(value)
        if reaction_family:
            value = self.reaction_family_priors.get(reaction_family)
            if value is not None:
                votes.append(value)
        if rule_source:
            value = self.rule_source_priors.get(rule_source)
            if value is not None:
                votes.append(value)
        if not votes:
            return float(self.default_prior)
        return float(sum(votes) / len(votes))


def _load_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in GENERATION_MEMORY_ARTIFACTS:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False).copy()
        if df.empty or "smiles" not in df.columns:
            continue
        frames.append(df.assign(artifact_name=path.stem))
    return frames


def _row_success_score(df: pd.DataFrame) -> pd.Series:
    final_pct = _series(df, "final_score", 0.0).rank(method="average", pct=True, ascending=True)
    verified_pct = _series(df, "verified_reward", _series(df, "final_score", 0.0)).rank(method="average", pct=True, ascending=True)
    feasibility = _series(df, "feasibility_score", 0.55).clip(lower=0.0, upper=1.0)
    crossdb = _series(df, "cross_database_consensus_score", 0.0).clip(lower=0.0, upper=1.0)
    external = _series(df, "external_evidence_support", 0.0).clip(lower=0.0, upper=1.0)
    readiness = _series(df, "experimental_readiness_score", 0.0).clip(lower=0.0, upper=1.0)
    generator_priority = _series(df, "generator_priority_score", 0.50).clip(lower=0.0, upper=1.0)
    structural = _series(df, "docking_rescore", 0.0).clip(lower=0.0, upper=1.0)
    low_risk = (1.0 - _series(df, "reward_hacking_risk", 0.20)).clip(lower=0.0, upper=1.0)
    parent_improvement = (
        0.50 * _series(df, "improved_over_parent_final_score", 0.0)
        + 0.30 * _series(df, "improved_over_parent_verified_reward", 0.0)
        + 0.20 * _series(df, "improved_over_parent_qed", 0.0)
    ).clip(lower=0.0, upper=1.0)
    audit_support = _series(df, "audit_status", 0.0)
    audit_pass = (audit_support.astype(str) == "pass").astype(float)
    hard_gate = _series(df, "hard_constraint_pass", 1.0).clip(lower=0.0, upper=1.0)

    score = (
        0.17 * final_pct
        + 0.15 * verified_pct
        + 0.12 * feasibility
        + 0.11 * crossdb
        + 0.09 * external
        + 0.08 * readiness
        + 0.08 * generator_priority
        + 0.08 * structural
        + 0.05 * low_risk
        + 0.04 * parent_improvement
        + 0.02 * audit_pass
        + 0.01 * hard_gate
    )
    return score.clip(lower=0.0, upper=1.0)


def _aggregate_priors(df: pd.DataFrame, key: str) -> dict[str, float]:
    if key not in df.columns:
        return {}
    grouped = (
        df.dropna(subset=[key])
        .groupby(key, dropna=True)
        .agg(
            count=("smiles", "count"),
            mean_success=("transformation_success_score", "mean"),
            pass_rate=("audit_pass_memory", "mean"),
        )
        .reset_index()
    )
    priors: dict[str, float] = {}
    for _, row in grouped.iterrows():
        count = max(1.0, float(row["count"]))
        shrinkage = min(1.0, count / 18.0)
        prior = 0.50 + shrinkage * ((0.72 * float(row["mean_success"]) + 0.28 * float(row["pass_rate"])) - 0.50)
        priors[str(row[key])] = _clip01(prior)
    return priors


@lru_cache(maxsize=1)
def load_transformation_memory() -> TransformationMemory:
    frames = _load_frames()
    if not frames:
        return TransformationMemory({}, {}, {}, {}, default_prior=0.50)

    df = pd.concat(frames, ignore_index=True, sort=False).copy()
    df = df.assign(
        transformation_success_score=_row_success_score(df),
        audit_pass_memory=(df.get("audit_status", pd.Series("review", index=df.index)).astype(str) == "pass").astype(float),
    )

    return TransformationMemory(
        action_name_priors=_aggregate_priors(df, "action_name"),
        category_priors=_aggregate_priors(df, "action_category"),
        reaction_family_priors=_aggregate_priors(df, "reaction_family"),
        rule_source_priors=_aggregate_priors(df, "action_rule_source"),
        default_prior=0.50,
    )
