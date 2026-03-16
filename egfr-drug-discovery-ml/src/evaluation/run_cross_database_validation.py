from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _plot_consensus_vs_readiness(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "cross_database_consensus_score" not in df.columns:
        return
    readiness = df["experimental_readiness_score"] if "experimental_readiness_score" in df.columns else pd.Series(0.0, index=df.index)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["cross_database_consensus_score"],
        readiness,
        c=df.get("predicted_pIC50", pd.Series(0.0, index=df.index)),
        cmap="viridis",
        alpha=0.72,
        s=28,
    )
    ax.axvline(0.55, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.axhline(0.70, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Cross-database consensus score")
    ax.set_ylabel("Experimental readiness score")
    ax.set_title("Cross-Database Support vs Experimental Readiness")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Predicted pIC50")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_status_counts(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "cross_database_status" not in df.columns:
        return
    counts = df["cross_database_status"].value_counts().reindex(["strong", "moderate", "weak"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_ylabel("Candidate count")
    ax.set_title("Cross-Database Validation Strength")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_external_evidence(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "external_evidence_support" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["external_evidence_support"],
        df.get("predicted_pIC50", pd.Series(0.0, index=df.index)),
        c=df.get("cross_database_consensus_score", pd.Series(0.0, index=df.index)),
        cmap="viridis",
        alpha=0.72,
        s=28,
    )
    ax.axvline(0.55, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("External evidence support")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("External Evidence Agent Support")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Cross-database consensus")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate candidates against multiple independent public databases.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv"),
    )
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    validator = CrossDatabaseValidator()
    out = validator.validate_frame(df)
    out = add_external_evidence_agent_ranking(out)
    out = out.sort_values(
        ["external_evidence_priority" if "external_evidence_priority" in out.columns else "cross_database_priority", "cross_database_consensus_score", "predicted_pIC50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["cross_database_rank"] = out.index + 1

    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "n_candidates": int(len(out)),
        "mean_consensus_score": float(out["cross_database_consensus_score"].mean()) if not out.empty else 0.0,
        "mean_independent_support_count": float(out["cross_database_independent_support_count"].mean()) if not out.empty else 0.0,
        "strong_rate": float((out["cross_database_status"] == "strong").mean()) if not out.empty else 0.0,
        "moderate_rate": float((out["cross_database_status"] == "moderate").mean()) if not out.empty else 0.0,
        "weak_rate": float((out["cross_database_status"] == "weak").mean()) if not out.empty else 0.0,
        "mean_external_evidence_support": float(out["external_evidence_support"].mean()) if "external_evidence_support" in out.columns and not out.empty else 0.0,
        "external_evidence_pass_rate": float((out["external_evidence_status"] == "pass").mean()) if "external_evidence_status" in out.columns and not out.empty else 0.0,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    consensus_plot = out_path.parent / "cross_database_consensus_vs_readiness.png"
    status_plot = out_path.parent / "cross_database_status_counts.png"
    external_plot = out_path.parent / "external_evidence_support_vs_potency.png"
    _plot_consensus_vs_readiness(out, consensus_plot)
    _plot_status_counts(out, status_plot)
    _plot_external_evidence(out, external_plot)

    print(f"[OK] Saved cross-database validation file: {out_path}")
    print(f"[OK] Saved cross-database summary: {summary_path}")
    preview_cols = [
        "smiles",
        "predicted_pIC50",
        "cross_database_consensus_score",
        "external_evidence_support",
        "cross_database_independent_support_count",
        "cross_database_status",
        "experimental_readiness_score",
    ]
    preview_cols = [column for column in preview_cols if column in out.columns]
    print(out[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
