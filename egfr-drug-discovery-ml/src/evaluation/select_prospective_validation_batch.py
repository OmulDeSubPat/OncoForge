from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.agents.multi_agent import resolve_priority_score_column
from src.agents.structure_evidence_arbiter import add_structure_evidence_arbiter
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark
from src.utils.similarity import morgan_fp, tanimoto_similarity


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _series(df: pd.DataFrame, column: str, default: float | pd.Series = 0.0) -> pd.Series:
    if isinstance(default, pd.Series):
        default_series = pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
    else:
        default_series = pd.Series(float(default), index=df.index, dtype=float)
    if column not in df.columns:
        return default_series
    return pd.to_numeric(df[column], errors="coerce").fillna(default_series)


def _prepare_source(df: pd.DataFrame | None, source_name: str, market_df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "experimental_readiness_score" not in out.columns:
        out = add_experimental_readiness(out, market_df=market_df)
    out["candidate_source"] = source_name
    return out


def _plot_prospective_batch(all_candidates: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    if all_candidates.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        all_candidates["novelty_score"],
        all_candidates["experimental_readiness_score"],
        c=all_candidates["prospective_acquisition_score"],
        cmap="viridis",
        alpha=0.45,
        s=18,
    )
    if not selected.empty:
        ax.scatter(
            selected["novelty_score"],
            selected["experimental_readiness_score"],
            c="#d62828",
            edgecolor="black",
            linewidth=0.5,
            s=40,
            label="Selected prospective batch",
        )
    ax.axhline(0.70, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Novelty score")
    ax.set_ylabel("Experimental readiness score")
    ax.set_title("Prospective Validation Batch: Exploration vs Readiness")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Prospective acquisition score")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "prospective_batch_readiness_vs_novelty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_prospective_validation_batch(
    batch_size: int = 18,
    similarity_threshold: float = 0.72,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    market_df = load_market_benchmark()

    sources = [
        ("optimized_readiness", _optional_csv(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv")),
        ("optimized_crossdb", _optional_csv(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv")),
        ("optimized_feasibility", _optional_csv(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv")),
        ("diverse", _optional_csv(PROJECT_ROOT / "reports" / "final_diverse_candidates.csv")),
        ("shortlist", _optional_csv(PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv")),
        ("generated", _optional_csv(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_crossdb.csv")),
        ("rl", _optional_csv(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_top_candidates.csv")),
        ("gpu_rl_dqn", _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_dqn" / "gpu_rl_top_candidates.csv")),
        ("gpu_actor_critic", _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_actor_critic" / "gpu_actor_critic_top_candidates.csv")),
    ]

    prepared = [
        _prepare_source(df, source_name, market_df)
        for source_name, df in sources
        if df is not None and not df.empty
    ]
    if not prepared:
        raise FileNotFoundError("No candidate artifacts are available for prospective validation batch selection.")

    candidates = pd.concat(prepared, ignore_index=True, sort=False)
    if "smiles" not in candidates.columns:
        raise ValueError("Prospective candidate pool is missing the smiles column.")

    validator = CrossDatabaseValidator()
    candidates = validator.validate_frame(candidates)
    candidates = add_external_evidence_agent_ranking(candidates)
    candidates = add_experimental_readiness(candidates, market_df=market_df)
    candidates = add_evidence_arbiter_ranking(candidates)
    candidates = add_structure_evidence_arbiter(candidates)

    base_score_column = resolve_priority_score_column(candidates)
    candidates["base_priority_score"] = _series(candidates, base_score_column, 0.0)
    candidates["novelty_score"] = _series(candidates, "novelty_score", 0.0)
    candidates["uncertainty"] = _series(candidates, "uncertainty", 0.15)
    candidates["feasibility_score"] = _series(candidates, "feasibility_score", 0.0)
    candidates["structure_agent_support"] = _series(candidates, "structure_agent_support", _series(candidates, "docking_rescore", 0.0))
    candidates["source_support_score"] = _series(candidates, "source_support_score", 0.0)
    candidates["max_active_similarity"] = _series(candidates, "max_active_similarity", 0.0)
    candidates["reward_hacking_risk"] = _series(candidates, "reward_hacking_risk", 0.5)
    candidates["cross_database_consensus_score"] = _series(candidates, "cross_database_consensus_score", 0.0)
    candidates["cross_database_independent_support_count"] = _series(candidates, "cross_database_independent_support_count", 0.0)
    candidates["external_evidence_support"] = _series(candidates, "external_evidence_support", 0.0)
    candidates["evidence_arbiter_support"] = _series(candidates, "evidence_arbiter_support", 0.0)

    if "audit_status" in candidates.columns:
        candidates = candidates[candidates["audit_status"] != "fail"].copy()
    if "veto" in candidates.columns:
        candidates = candidates[candidates["veto"] == False].copy()
    if "experimental_readiness_status" in candidates.columns:
        candidates = candidates[candidates["experimental_readiness_status"] != "hold"].copy()
    if "cross_database_status" in candidates.columns:
        candidates = candidates[candidates["cross_database_status"] != "weak"].copy()
    if "structure_evidence_status" in candidates.columns:
        candidates = candidates[candidates["structure_evidence_status"] != "fail"].copy()

    candidates = candidates.sort_values(
        [
            "structure_evidence_state_priority" if "structure_evidence_state_priority" in candidates.columns else "experimental_readiness_priority",
            "structure_evidence_pareto_front_rank" if "structure_evidence_pareto_front_rank" in candidates.columns else "experimental_readiness_priority",
            "structure_evidence_priority" if "structure_evidence_priority" in candidates.columns else "base_priority_score",
            "base_priority_score",
        ],
        ascending=[True, True, False, False],
    ).drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)

    base_percentile = candidates["base_priority_score"].rank(method="average", pct=True, ascending=True)
    uncertainty_target = 0.18
    uncertainty_window = (1.0 - ((_series(candidates, "uncertainty", uncertainty_target) - uncertainty_target).abs() / uncertainty_target)).clip(lower=0.0, upper=1.0)
    validation_evidence = (
        0.40 * _series(candidates, "experimental_readiness_score", 0.0)
        + 0.20 * _series(candidates, "feasibility_score", 0.0)
        + 0.15 * _series(candidates, "cross_database_consensus_score", 0.0)
        + 0.08 * _series(candidates, "external_evidence_support", 0.0)
        + 0.07 * _series(candidates, "evidence_arbiter_support", 0.0)
        + 0.10 * _series(candidates, "source_support_score", 0.0)
        + 0.10 * _series(candidates, "max_active_similarity", 0.0)
        + 0.05 * (_series(candidates, "cross_database_independent_support_count", 0.0) / 3.0).clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    exploration_score = (
        0.60 * _series(candidates, "novelty_score", 0.0)
        + 0.40 * uncertainty_window
    ).clip(lower=0.0, upper=1.0)
    source_bonus = candidates["candidate_source"].map(
        {
            "shortlist": 0.05,
            "rl": 0.04,
            "diverse": 0.03,
            "optimized_readiness": 0.03,
            "optimized_crossdb": 0.03,
            "optimized_feasibility": 0.02,
            "generated": 0.02,
            "gpu_rl_dqn": 0.03,
            "gpu_actor_critic": 0.03,
        }
    ).fillna(0.0)
    candidates["prospective_validation_evidence"] = validation_evidence
    candidates["prospective_exploration_score"] = exploration_score
    candidates["prospective_acquisition_score"] = (
        0.38 * _series(candidates, "experimental_readiness_score", 0.0)
        + 0.18 * _series(candidates, "structure_agent_support", 0.0)
        + 0.14 * _series(candidates, "cross_database_consensus_score", 0.0)
        + 0.08 * _series(candidates, "external_evidence_support", 0.0)
        + 0.08 * _series(candidates, "evidence_arbiter_support", 0.0)
        + 0.10 * _series(candidates, "structure_evidence_support", 0.0)
        + 0.15 * base_percentile
        + 0.10 * exploration_score
        + 0.10 * validation_evidence
        + 0.10 * _series(candidates, "structure_evidence_pareto_priority_bonus", 0.0)
        + source_bonus
    )

    candidates = candidates.sort_values(
        ["prospective_acquisition_score", "experimental_readiness_priority", "predicted_pIC50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_rows = []
    selected_fps = []
    source_counts: dict[str, int] = {}
    max_per_source = max(4, batch_size // 2)

    for _, row in candidates.iterrows():
        source_name = str(row.get("candidate_source", "unknown"))
        if source_counts.get(source_name, 0) >= max_per_source:
            continue
        fp = morgan_fp(smiles=row["smiles"])
        if fp is None:
            continue
        too_similar = any(tanimoto_similarity(fp, prev_fp) >= similarity_threshold for prev_fp in selected_fps)
        if too_similar:
            continue
        selected_rows.append(row.to_dict())
        selected_fps.append(fp)
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        if len(selected_rows) >= batch_size:
            break

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected = selected.sort_values(
            ["prospective_acquisition_score", "experimental_readiness_priority", "predicted_pIC50"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected["prospective_batch_rank"] = selected.index + 1

    summary = {
        "candidate_pool_size": int(len(candidates)),
        "selected_batch_size": int(len(selected)),
        "batch_sources": selected["candidate_source"].value_counts().to_dict() if not selected.empty else {},
        "batch_status_counts": selected["experimental_readiness_status"].value_counts().to_dict() if "experimental_readiness_status" in selected.columns and not selected.empty else {},
        "mean_acquisition_score": float(selected["prospective_acquisition_score"].mean()) if not selected.empty else 0.0,
        "mean_readiness_score": float(selected["experimental_readiness_score"].mean()) if not selected.empty else 0.0,
        "mean_feasibility_score": float(selected["feasibility_score"].mean()) if "feasibility_score" in selected.columns and not selected.empty else 0.0,
        "mean_docking_rescore": float(selected["docking_rescore"].mean()) if "docking_rescore" in selected.columns and not selected.empty else 0.0,
        "mean_external_evidence_support": float(selected["external_evidence_support"].mean()) if "external_evidence_support" in selected.columns and not selected.empty else 0.0,
        "mean_evidence_arbiter_support": float(selected["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in selected.columns and not selected.empty else 0.0,
        "mean_structure_evidence_support": float(selected["structure_evidence_support"].mean()) if "structure_evidence_support" in selected.columns and not selected.empty else 0.0,
        "pareto_front_rate": float(selected["structure_evidence_pareto_is_front"].mean()) if "structure_evidence_pareto_is_front" in selected.columns and not selected.empty else 0.0,
    }

    target_path = out_path or (PROJECT_ROOT / "reports" / "prospective_validation_batch.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(target_path, index=False)

    summary_path = target_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_prospective_batch(candidates, selected, target_path.parent)
    return selected, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a prospective validation batch using active-learning style acquisition.")
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "prospective_validation_batch.csv"),
    )
    args = parser.parse_args(argv)

    out_path = _resolve_path(args.out)
    selected, summary = build_prospective_validation_batch(
        batch_size=max(6, int(args.batch_size)),
        similarity_threshold=float(args.similarity_threshold),
        out_path=out_path,
    )
    print(f"[OK] Saved prospective validation batch: {out_path}")
    print(f"[OK] Saved batch summary: {out_path.with_suffix('.summary.json')}")
    print(json.dumps(summary, indent=2))
    preview_cols = [
        "prospective_batch_rank",
        "candidate_source",
        "smiles",
        "predicted_pIC50",
        "experimental_readiness_score",
        "prospective_acquisition_score",
        "structure_evidence_support",
        "experimental_readiness_status",
        "experimental_track",
    ]
    preview_cols = [column for column in preview_cols if column in selected.columns]
    print(selected[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
