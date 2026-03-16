from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.agents.multi_agent import build_default_scorer, resolve_priority_score_column, score_smiles_list
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.assessor import FeasibilityAssessor
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark
from src.structure.docking_rescoring import StructuralConsensusRescorer
from src.structure.interaction_analysis import PoseInteractionAnalyzer
from src.utils.chem import canonicalize_smiles


def _safe_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _plot_recall(curves_df: pd.DataFrame, out_path: Path) -> None:
    if curves_df.empty:
        return
    plot_df = curves_df.copy()
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, plot_df["protected_recall"], width, label="Protected", color="#2a9d8f")
    ax.bar(x + width / 2, plot_df["naive_recall"], width, label="Naive", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {int(k)}" for k in plot_df["k"]])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rediscovery recall")
    ax.set_title("Known EGFR Reference Recovery in a Hard Candidate Panel")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_shift(panel_df: pd.DataFrame, out_path: Path) -> None:
    positive_df = panel_df[panel_df["benchmark_positive"] == True].copy()
    if positive_df.empty:
        return
    positive_df["rank_shift"] = positive_df["naive_panel_rank"] - positive_df["protected_panel_rank"]
    positive_df = positive_df.sort_values("rank_shift", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(positive_df["benchmark_name"], positive_df["rank_shift"], color="#457b9d")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_ylabel("Naive rank - protected rank")
    ax.set_title("Protected Ranking Shift for Reference EGFR Positives")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _enrich_with_structure(df: pd.DataFrame, pose_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "docking_rescore" in out.columns and out["docking_rescore"].notna().all():
        return out
    rescorer = StructuralConsensusRescorer(
        backend="auto",
        pose_dir=pose_dir,
        vina_cpu=1,
        vina_exhaustiveness=6,
        vina_num_modes=5,
    )
    analyzer = PoseInteractionAnalyzer()
    rows = []
    for idx, (_, row) in enumerate(out.iterrows(), start=1):
        out_row = row.to_dict()
        if pd.isna(out_row.get("docking_rescore", np.nan)) and rescorer.is_available():
            rescored = rescorer.score_smiles(str(out_row["smiles"]), ligand_name=f"rediscovery_{idx:03d}")
            out_row.update(rescored)
            pose_path = rescored.get("docking_pose_path")
            if isinstance(pose_path, str) and pose_path:
                out_row.update(analyzer.analyze_pose(pose_path, smiles=str(out_row["smiles"])))
        rows.append(out_row)
    return pd.DataFrame(rows)


def _enrich_panel(df: pd.DataFrame, pose_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _enrich_with_structure(df, pose_dir=pose_dir)
    assessor = FeasibilityAssessor()
    rows = []
    for _, row in out.iterrows():
        out_row = row.to_dict()
        if "feasibility_score" not in out.columns or pd.isna(out_row.get("feasibility_score", np.nan)):
            feasibility = assessor.assess(
                str(out_row["smiles"]),
                synthetic_feasibility_score=float(out_row["synthetic_feasibility_score"]) if pd.notna(out_row.get("synthetic_feasibility_score", np.nan)) else None,
                medchem_realism_score=float(out_row["medchem_realism_score"]) if pd.notna(out_row.get("medchem_realism_score", np.nan)) else None,
                transformation_confidence=float(out_row["transformation_confidence_score"]) if pd.notna(out_row.get("transformation_confidence_score", np.nan)) else None,
                reaction_family=str(out_row["reaction_family"]) if pd.notna(out_row.get("reaction_family", np.nan)) else None,
                docking_rescore=float(out_row["docking_rescore"]) if pd.notna(out_row.get("docking_rescore", np.nan)) else None,
                interaction_support_score=float(out_row["interaction_support_score"]) if pd.notna(out_row.get("interaction_support_score", np.nan)) else None,
                interaction_key_residue_count=int(out_row["interaction_key_residue_count"]) if pd.notna(out_row.get("interaction_key_residue_count", np.nan)) else None,
            )
            out_row.update(feasibility)
        rows.append(out_row)
    out = pd.DataFrame(rows)
    out = CrossDatabaseValidator().validate_frame(out)
    out = add_external_evidence_agent_ranking(out)
    out = add_experimental_readiness(out, market_df=load_market_benchmark(), sort_output=False)
    out = add_evidence_arbiter_ranking(out)
    return out


def _load_market_positives() -> pd.DataFrame:
    market_path = PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv"
    if not market_path.exists():
        market_path = PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv"
    market_df = pd.read_csv(market_path, low_memory=False)
    market_df = market_df.copy()
    market_df["benchmark_name"] = market_df.get("name", pd.Series("marketed", index=market_df.index)).astype(str)
    market_df["benchmark_source"] = "marketed"
    market_df["benchmark_positive"] = True
    return market_df


def _load_iuphar_positives(max_iuphar: int) -> pd.DataFrame:
    iuphar_path = PROJECT_ROOT / "data" / "processed" / "iuphar_egfr_reference.csv"
    ref_df = pd.read_csv(iuphar_path, low_memory=False)
    ref_df = ref_df[pd.to_numeric(ref_df["pIC50_median"], errors="coerce") >= 8.5].copy()
    ref_df["smiles"] = ref_df["smiles"].map(canonicalize_smiles)
    ref_df = ref_df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
    ref_df = ref_df.sort_values("pIC50_median", ascending=False).head(int(max_iuphar)).copy()
    if ref_df.empty:
        return ref_df
    scorer = build_default_scorer()
    scored = score_smiles_list(ref_df["smiles"].tolist(), scorer=scorer)
    ref_df = ref_df.rename(columns={"pIC50_median": "benchmark_reference_pIC50"})
    scored = scored.merge(
        ref_df[["smiles", "ligand_name", "benchmark_reference_pIC50"]],
        on="smiles",
        how="left",
    )
    scored["benchmark_name"] = scored["ligand_name"].fillna("iuphar_reference").astype(str)
    scored["benchmark_source"] = "iuphar"
    scored["benchmark_positive"] = True
    return scored.drop(columns=[column for column in ["ligand_name"] if column in scored.columns], errors="ignore")


def _load_challenger_pool(max_rows: int, positive_smiles: set[str]) -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "reports" / "final_diverse_candidates.csv",
        PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv",
        PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_crossdb.csv",
    ]
    frames = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "smiles" in df.columns:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["smiles"])
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["smiles"]).copy()
    pool = pool[~pool["smiles"].isin(positive_smiles)].copy()
    if "cross_database_status" in pool.columns:
        pool = pool[pool["cross_database_status"].isin(["moderate", "strong"])].copy()
    if "audit_status" in pool.columns:
        pool = pool[pool["audit_status"] != "fail"].copy()
    sort_col = "evidence_arbiter_priority" if "evidence_arbiter_priority" in pool.columns else resolve_priority_score_column(pool)
    pool = pool.sort_values(sort_col, ascending=False).head(int(max_rows)).copy()
    pool["benchmark_name"] = "challenger_" + pool.index.astype(str)
    pool["benchmark_source"] = "challenger"
    pool["benchmark_positive"] = False
    return pool


def _recall_at_k(df: pd.DataFrame, rank_col: str, k: int) -> float:
    positives = df[df["benchmark_positive"] == True].copy()
    if positives.empty:
        return 0.0
    return float((positives[rank_col] <= int(k)).mean())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a rediscovery benchmark against marketed and IUPHAR EGFR positives.")
    parser.add_argument("--max-iuphar", type=int, default=12, help="Maximum number of strong IUPHAR references to include.")
    parser.add_argument("--challenger-multiplier", type=int, default=8, help="Number of challenger molecules per positive reference.")
    args = parser.parse_args(argv)

    market_df = _load_market_positives()
    iuphar_df = _load_iuphar_positives(max_iuphar=int(args.max_iuphar))
    positive_df = pd.concat([market_df, iuphar_df], ignore_index=True, sort=False)
    positive_df["smiles"] = positive_df["smiles"].map(canonicalize_smiles)
    positive_df = positive_df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    positive_smiles = set(positive_df["smiles"].tolist())
    challenger_df = _load_challenger_pool(
        max_rows=max(24, int(len(positive_df) * int(args.challenger_multiplier))),
        positive_smiles=positive_smiles,
    )

    panel = pd.concat([positive_df, challenger_df], ignore_index=True, sort=False).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    panel = _enrich_panel(panel, pose_dir=PROJECT_ROOT / "reports" / "vina_poses" / "rediscovery_panel")

    base_priority_col = resolve_priority_score_column(panel)
    panel["rediscovery_priority_score"] = (
        0.55 * _safe_numeric(panel, "predicted_pIC50", 0.0)
        + 1.20 * _safe_numeric(panel, "external_evidence_support", 0.0)
        + 0.95 * _safe_numeric(panel, "evidence_arbiter_support", 0.0)
        + 0.85 * _safe_numeric(panel, "cross_database_consensus_score", 0.0)
        + 0.80 * _safe_numeric(panel, "docking_rescore", 0.0)
        + 0.60 * _safe_numeric(panel, "interaction_support_score", 0.0)
        + 0.40 * _safe_numeric(panel, "marketed_support_score", 0.0)
        + 0.35 * _safe_numeric(panel, "feasibility_score", 0.0)
        + 0.25 * _safe_numeric(panel, "experimental_readiness_score", 0.0)
        + 0.10 * (_safe_numeric(panel, "cross_database_independent_support_count", 0.0) / 4.0)
    )
    protected_rank_df = panel.sort_values(
        [
            "evidence_arbiter_state_priority" if "evidence_arbiter_state_priority" in panel.columns else "audit_priority",
            "rediscovery_priority_score",
            "evidence_arbiter_priority" if "evidence_arbiter_priority" in panel.columns else base_priority_col,
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    protected_rank_df["protected_panel_rank"] = np.arange(1, len(protected_rank_df) + 1)

    naive_rank_df = protected_rank_df.sort_values("naive_score" if "naive_score" in protected_rank_df.columns else base_priority_col, ascending=False).reset_index(drop=True)
    naive_rank_df["naive_panel_rank"] = np.arange(1, len(naive_rank_df) + 1)
    panel = protected_rank_df.merge(
        naive_rank_df[["smiles", "naive_panel_rank"]],
        on="smiles",
        how="left",
    )
    panel["rank_shift"] = panel["naive_panel_rank"] - panel["protected_panel_rank"]

    ks = [5, 10, 20, 50]
    recall_rows = []
    for k in ks:
        recall_rows.append(
            {
                "k": int(k),
                "protected_recall": _recall_at_k(panel, "protected_panel_rank", k),
                "naive_recall": _recall_at_k(panel, "naive_panel_rank", k),
            }
        )
    recall_df = pd.DataFrame(recall_rows)

    positive_panel = panel[panel["benchmark_positive"] == True].copy()
    summary = {
        "panel_size": int(len(panel)),
        "positive_count": int(len(positive_panel)),
        "challenger_count": int((panel["benchmark_positive"] == False).sum()),
        "protected_median_positive_rank": float(positive_panel["protected_panel_rank"].median()) if not positive_panel.empty else None,
        "naive_median_positive_rank": float(positive_panel["naive_panel_rank"].median()) if not positive_panel.empty else None,
        "protected_top10_recall": float(recall_df.loc[recall_df["k"] == 10, "protected_recall"].iloc[0]) if not recall_df.empty else 0.0,
        "naive_top10_recall": float(recall_df.loc[recall_df["k"] == 10, "naive_recall"].iloc[0]) if not recall_df.empty else 0.0,
        "protected_top20_recall": float(recall_df.loc[recall_df["k"] == 20, "protected_recall"].iloc[0]) if not recall_df.empty else 0.0,
        "naive_top20_recall": float(recall_df.loc[recall_df["k"] == 20, "naive_recall"].iloc[0]) if not recall_df.empty else 0.0,
        "positive_audit_pass_rate": float((positive_panel.get("audit_status", pd.Series("review", index=positive_panel.index)) == "pass").mean()) if not positive_panel.empty else 0.0,
        "positive_external_evidence_pass_rate": float((positive_panel.get("external_evidence_status", pd.Series("review", index=positive_panel.index)) == "pass").mean()) if not positive_panel.empty else 0.0,
        "positive_arbiter_pass_rate": float((positive_panel.get("evidence_arbiter_status", pd.Series("review", index=positive_panel.index)) == "pass").mean()) if not positive_panel.empty else 0.0,
        "positive_sources": positive_panel["benchmark_source"].value_counts().to_dict(),
    }

    out_dir = PROJECT_ROOT / "reports" / "rediscovery_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = out_dir / "rediscovery_panel.csv"
    summary_json = out_dir / "rediscovery_summary.json"
    recall_csv = out_dir / "rediscovery_recall_at_k.csv"
    recall_plot = out_dir / "rediscovery_recall_at_k.png"
    rank_shift_plot = out_dir / "rediscovery_rank_shift.png"
    positives_csv = out_dir / "rediscovery_positive_controls.csv"

    panel.to_csv(panel_csv, index=False)
    positive_panel.to_csv(positives_csv, index=False)
    recall_df.to_csv(recall_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_recall(recall_df, recall_plot)
    _plot_rank_shift(panel, rank_shift_plot)

    print(f"[OK] Saved rediscovery panel: {panel_csv}")
    print(f"[OK] Saved rediscovery summary: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
