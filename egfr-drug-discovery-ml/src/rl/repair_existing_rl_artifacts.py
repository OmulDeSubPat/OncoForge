from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.structure_evidence_arbiter import add_structure_evidence_arbiter
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.assessor import FeasibilityAssessor
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _reassess_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    assessor = FeasibilityAssessor()
    rows: list[dict] = []
    for _, row in df.iterrows():
        feasibility = assessor.assess(
            str(row["smiles"]),
            action_name=str(row["action_name"]) if "action_name" in row and pd.notna(row["action_name"]) else None,
            action_rule_source=str(row["action_rule_source"]) if "action_rule_source" in row and pd.notna(row["action_rule_source"]) else None,
            synthetic_route=str(row["synthetic_route"]) if "synthetic_route" in row and pd.notna(row["synthetic_route"]) else None,
            synthetic_feasibility_score=float(row["synthetic_feasibility_score"]) if "synthetic_feasibility_score" in row and pd.notna(row["synthetic_feasibility_score"]) else None,
            medchem_realism_score=float(row["medchem_realism_score"]) if "medchem_realism_score" in row and pd.notna(row["medchem_realism_score"]) else None,
            transformation_confidence=float(row["transformation_confidence_score"]) if "transformation_confidence_score" in row and pd.notna(row["transformation_confidence_score"]) else None,
            reaction_family=str(row["reaction_family"]) if "reaction_family" in row and pd.notna(row["reaction_family"]) else None,
            docking_rescore=float(row["docking_rescore"]) if "docking_rescore" in row and pd.notna(row["docking_rescore"]) else None,
            interaction_support_score=float(row["interaction_support_score"]) if "interaction_support_score" in row and pd.notna(row["interaction_support_score"]) else None,
            interaction_key_residue_count=int(row["interaction_key_residue_count"]) if "interaction_key_residue_count" in row and pd.notna(row["interaction_key_residue_count"]) else None,
        )
        updated = row.to_dict()
        updated.update(feasibility)
        rows.append(updated)
    return pd.DataFrame(rows)


def _enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    enriched = CrossDatabaseValidator().validate_frame(df)
    enriched = add_experimental_readiness(enriched, market_df=load_market_benchmark(), sort_output=False)
    enriched = add_evidence_arbiter_ranking(enriched)
    enriched = add_structure_evidence_arbiter(enriched)
    return enriched


def _repair_verifiable_rl() -> None:
    out_dir = PROJECT_ROOT / "reports" / "rl_verifiable"
    csv_path = out_dir / "rl_top_candidates.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = _enrich_frame(_reassess_frame(df))
    df["rl_priority_score"] = (
        _series(df, "verified_reward", 0.0)
        + 0.80 * _series(df, "feasibility_score", 0.0)
        + 0.45 * _series(df, "docking_rescore", 0.0)
        + 0.45 * _series(df, "interaction_support_score", 0.0)
        + 0.30 * _series(df, "cross_database_consensus_score", 0.0)
        + 0.30 * _series(df, "external_evidence_support", 0.0)
        + 0.25 * _series(df, "experimental_readiness_score", 0.0)
        + 0.25 * _series(df, "evidence_arbiter_support", 0.0)
        + 0.30 * _series(df, "structure_evidence_support", 0.0)
        + 0.12 * _series(df, "structure_evidence_guardrail", 0.0)
        + 0.16 * _series(df, "adaptive_action_prior", 0.5)
    )
    df["rl_audit_priority"] = df.get("audit_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df["rl_external_priority"] = df.get("external_evidence_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df["rl_readiness_priority"] = df.get("experimental_readiness_status", pd.Series("supporting", index=df.index)).map({"ready": 0, "supporting": 1, "hold": 2}).fillna(1).astype(int)
    df["rl_arbiter_priority"] = df.get("evidence_arbiter_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df["rl_structure_priority"] = df.get("structure_evidence_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df = df.sort_values(
        [
            "rl_structure_priority",
            "rl_arbiter_priority",
            "rl_audit_priority",
            "rl_external_priority",
            "rl_readiness_priority",
            "rl_priority_score",
            "docking_rescore",
            "predicted_pIC50",
            "QED",
        ],
        ascending=[True, True, True, True, True, False, False, False, False],
    ).reset_index(drop=True)
    df["rl_rank"] = df.index + 1
    df.to_csv(csv_path, index=False)
    (out_dir / "rl_top_candidates_crossdb.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary_path = out_dir / "rl_training_summary.json"
    summary = _load_json(summary_path)
    summary.update(
        {
            "mean_adaptive_action_prior": float(_series(df, "adaptive_action_prior", 0.0).mean()) if not df.empty else 0.0,
            "mean_cross_database_consensus": float(_series(df, "cross_database_consensus_score", 0.0).mean()) if not df.empty else 0.0,
            "external_evidence_pass_rate": float((df["external_evidence_status"] == "pass").mean()) if "external_evidence_status" in df.columns and not df.empty else 0.0,
            "mean_external_evidence_support": float(_series(df, "external_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_experimental_readiness_score": float(_series(df, "experimental_readiness_score", 0.0).mean()) if not df.empty else 0.0,
            "readiness_ready_rate": float((df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in df.columns and not df.empty else 0.0,
            "arbiter_pass_rate": float((df["evidence_arbiter_status"] == "pass").mean()) if "evidence_arbiter_status" in df.columns and not df.empty else 0.0,
            "mean_evidence_arbiter_support": float(_series(df, "evidence_arbiter_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_support": float(_series(df, "structure_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_guardrail": float(_series(df, "structure_evidence_guardrail", 0.0).mean()) if not df.empty else 0.0,
            "structure_evidence_pass_rate": float((df["structure_evidence_status"] == "pass").mean()) if "structure_evidence_status" in df.columns and not df.empty else 0.0,
            "top_candidate": df.head(1).to_dict(orient="records"),
        }
    )
    _write_json(summary_path, summary)


def _repair_gpu_dqn() -> None:
    out_dir = PROJECT_ROOT / "reports" / "rl_gpu_dqn"
    csv_path = out_dir / "gpu_rl_top_candidates.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = _enrich_frame(_reassess_frame(df))
    df["gpu_rl_priority_score"] = (
        _series(df, "verified_reward", 0.0)
        + 0.80 * _series(df, "feasibility_score", 0.0)
        + 0.40 * _series(df, "docking_rescore", 0.0)
        + 0.35 * _series(df, "interaction_support_score", 0.0)
        + 0.30 * _series(df, "cross_database_consensus_score", 0.0)
        + 0.30 * _series(df, "external_evidence_support", 0.0)
        + 0.28 * _series(df, "experimental_readiness_score", 0.0)
        + 0.30 * _series(df, "evidence_arbiter_support", 0.0)
        + 0.32 * _series(df, "structure_evidence_support", 0.0)
        + 0.12 * _series(df, "structure_evidence_guardrail", 0.0)
        + 0.16 * _series(df, "adaptive_action_prior", 0.5)
    )
    df["gpu_rl_arbiter_priority"] = df.get("evidence_arbiter_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df["gpu_rl_structure_priority"] = df.get("structure_evidence_status", pd.Series("review", index=df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    df = df.sort_values(
        ["gpu_rl_structure_priority", "gpu_rl_arbiter_priority", "gpu_rl_priority_score", "predicted_pIC50"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    df["gpu_rl_rank"] = df.index + 1
    df.to_csv(csv_path, index=False)

    summary_path = out_dir / "gpu_rl_training_summary.json"
    summary = _load_json(summary_path)
    summary.update(
        {
            "mean_adaptive_action_prior": float(_series(df, "adaptive_action_prior", 0.0).mean()) if not df.empty else 0.0,
            "mean_cross_database_consensus": float(_series(df, "cross_database_consensus_score", 0.0).mean()) if not df.empty else 0.0,
            "mean_external_evidence_support": float(_series(df, "external_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_experimental_readiness_score": float(_series(df, "experimental_readiness_score", 0.0).mean()) if not df.empty else 0.0,
            "ready_rate": float((df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in df.columns and not df.empty else 0.0,
            "mean_evidence_arbiter_support": float(_series(df, "evidence_arbiter_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_support": float(_series(df, "structure_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_guardrail": float(_series(df, "structure_evidence_guardrail", 0.0).mean()) if not df.empty else 0.0,
            "arbiter_pass_rate": float((df["evidence_arbiter_status"] == "pass").mean()) if "evidence_arbiter_status" in df.columns and not df.empty else 0.0,
            "structure_evidence_pass_rate": float((df["structure_evidence_status"] == "pass").mean()) if "structure_evidence_status" in df.columns and not df.empty else 0.0,
            "top_candidate": df.head(1).to_dict(orient="records"),
        }
    )
    _write_json(summary_path, summary)


def _repair_actor_critic() -> None:
    out_dir = PROJECT_ROOT / "reports" / "rl_gpu_actor_critic"
    csv_path = out_dir / "gpu_actor_critic_top_candidates.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = _enrich_frame(_reassess_frame(df))
    df["actor_critic_priority_score"] = (
        _series(df, "verified_reward", 0.0)
        + 0.85 * _series(df, "feasibility_score", 0.0)
        + 0.40 * _series(df, "docking_rescore", 0.0)
        + 0.35 * _series(df, "interaction_support_score", 0.0)
        + 0.35 * _series(df, "cross_database_consensus_score", 0.0)
        + 0.35 * _series(df, "external_evidence_support", 0.0)
        + 0.25 * _series(df, "experimental_readiness_score", 0.0)
        + 0.28 * _series(df, "structure_evidence_support", 0.0)
        + 0.12 * _series(df, "structure_evidence_guardrail", 0.0)
        + 0.20 * _series(df, "generator_priority_score", 0.0)
        + 0.18 * _series(df, "adaptive_action_prior", 0.5)
    )
    df = df.sort_values(
        [
            "structure_evidence_state_priority" if "structure_evidence_state_priority" in df.columns else "actor_critic_priority_score",
            "structure_evidence_pareto_front_rank" if "structure_evidence_pareto_front_rank" in df.columns else "actor_critic_priority_score",
            "structure_evidence_priority" if "structure_evidence_priority" in df.columns else "actor_critic_priority_score",
            "actor_critic_priority_score",
            "predicted_pIC50",
            "QED",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)
    df["actor_critic_rank"] = df.index + 1
    df.to_csv(csv_path, index=False)

    summary_path = out_dir / "gpu_actor_critic_summary.json"
    summary = _load_json(summary_path)
    summary.update(
        {
            "mean_feasibility_score": float(_series(df, "feasibility_score", 0.0).mean()) if not df.empty else 0.0,
            "mean_cross_database_consensus": float(_series(df, "cross_database_consensus_score", 0.0).mean()) if not df.empty else 0.0,
            "mean_external_evidence_support": float(_series(df, "external_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_adaptive_action_prior": float(_series(df, "adaptive_action_prior", 0.0).mean()) if not df.empty else 0.0,
            "mean_experimental_readiness_score": float(_series(df, "experimental_readiness_score", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_support": float(_series(df, "structure_evidence_support", 0.0).mean()) if not df.empty else 0.0,
            "mean_structure_evidence_guardrail": float(_series(df, "structure_evidence_guardrail", 0.0).mean()) if not df.empty else 0.0,
            "ready_rate": float((df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in df.columns and not df.empty else 0.0,
            "structure_evidence_pass_rate": float((df["structure_evidence_status"] == "pass").mean()) if "structure_evidence_status" in df.columns and not df.empty else 0.0,
            "top_candidate": df.head(1).to_dict(orient="records"),
        }
    )
    _write_json(summary_path, summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair stale RL report artifacts without retraining the agents.")
    parser.parse_args(argv)

    _repair_verifiable_rl()
    _repair_gpu_dqn()
    _repair_actor_critic()
    print("[OK] Repaired existing RL artifacts.")


if __name__ == "__main__":
    main()
