from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "ml_chapter_figures"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_split_performance(model_summary: dict) -> None:
    labels = ["Random", "Scaffold", "Temporal"]
    mae = [
        model_summary["random_split"]["mae"],
        model_summary["scaffold_split"]["mae"],
        model_summary["temporal_split"]["mae"],
    ]
    rmse = [
        model_summary["random_split"]["rmse"],
        model_summary["scaffold_split"]["rmse"],
        model_summary["temporal_split"]["rmse"],
    ]
    r2 = [
        model_summary["random_split"]["r2"],
        model_summary["scaffold_split"]["r2"],
        model_summary["temporal_split"]["r2"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    colors = ["#2563EB", "#0EA5E9", "#F97316"]

    for ax, metric, title in zip(
        axes,
        [mae, rmse, r2],
        ["MAE pe split-uri", "RMSE pe split-uri", "R2 pe split-uri"],
    ):
        bars = ax.bar(labels, metric, color=colors, width=0.62)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title.startswith("R2"):
            ax.axhline(0.0, color="#9CA3AF", linewidth=1)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in metric], padding=3)

    fig.suptitle("Generalizarea modelului de potenta EGFR", fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "ml_01_split_performance.png")


def build_source_holdout_overview(source_holdout_df: pd.DataFrame) -> None:
    df = source_holdout_df.copy()
    label_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
        "chembl": "ChEMBL",
    }
    df["source_label"] = df["source"].map(label_map).fillna(df["source"])
    df = df.sort_values("recall_top20pct", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors = ["#14B8A6", "#60A5FA", "#F59E0B", "#EF4444"]

    bars_rmse = axes[0].bar(df["source_label"], df["rmse"], color=colors[: len(df)], width=0.62)
    axes[0].set_title("Eroare pe surse tinute complet in afara antrenarii", fontweight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].bar_label(bars_rmse, labels=[f"{value:.3f}" for value in df["rmse"]], padding=3)

    recall_values = (df["recall_top20pct"] * 100.0).tolist()
    bars_recall = axes[1].bar(df["source_label"], recall_values, color=colors[: len(df)], width=0.62)
    axes[1].set_title("Recall pentru molecule puternice in top 20%", fontweight="bold")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, max(recall_values) * 1.2)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].bar_label(bars_recall, labels=[f"{value:.1f}%" for value in recall_values], padding=3)

    fig.suptitle("Validare externa pe baze de date independente", fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "ml_02_source_holdout_overview.png")


def build_uncertainty_vs_error(predictions_df: pd.DataFrame) -> None:
    df = predictions_df.copy()
    df["predicted_uncertainty"] = pd.to_numeric(df["predicted_uncertainty"], errors="coerce")
    df["absolute_error"] = pd.to_numeric(df["absolute_error"], errors="coerce")
    df = df.dropna(subset=["predicted_uncertainty", "absolute_error"]).copy()

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    hb = ax.hexbin(
        df["predicted_uncertainty"],
        df["absolute_error"],
        gridsize=35,
        cmap="YlGnBu",
        mincnt=1,
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Numar de molecule")

    bins = np.quantile(df["predicted_uncertainty"], np.linspace(0.0, 1.0, 9))
    bins = np.unique(bins)
    if len(bins) > 2:
        grouped = (
            df.assign(bin=pd.cut(df["predicted_uncertainty"], bins=bins, include_lowest=True))
            .groupby("bin", observed=False)
            .agg(
                mean_uncertainty=("predicted_uncertainty", "mean"),
                mean_error=("absolute_error", "mean"),
            )
            .dropna()
        )
        if not grouped.empty:
            ax.plot(
                grouped["mean_uncertainty"],
                grouped["mean_error"],
                color="#DC2626",
                linewidth=2.2,
                marker="o",
                label="eroare medie pe intervale",
            )
            ax.legend(frameon=False, loc="upper left")

    ax.set_title("Incertitudine estimata vs eroare observata", fontweight="bold")
    ax.set_xlabel("Incertitudine predictiva")
    ax.set_ylabel("Eroare absoluta")
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "ml_03_uncertainty_vs_error.png")


def build_multi_agent_ablation(ablation_df: pd.DataFrame) -> None:
    df = ablation_df.copy()
    keep = ["naive_proxy", "verified_plus_mo", "protected_final"]
    df = df[df["strategy"].isin(keep)].copy()
    label_map = {
        "naive_proxy": "Naive proxy",
        "verified_plus_mo": "Verified + MO",
        "protected_final": "Protected final",
    }
    color_map = {
        "naive_proxy": "#EF4444",
        "verified_plus_mo": "#F59E0B",
        "protected_final": "#10B981",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for strategy in keep:
        subset = df[df["strategy"] == strategy].sort_values("top_k")
        axes[0].plot(
            subset["top_k"],
            subset["mean_reward_hacking_risk"],
            marker="o",
            linewidth=2.1,
            color=color_map[strategy],
            label=label_map[strategy],
        )
        axes[1].plot(
            subset["top_k"],
            subset["audit_pass_rate"] * 100.0,
            marker="o",
            linewidth=2.1,
            color=color_map[strategy],
            label=label_map[strategy],
        )

    axes[0].set_title("Risc mediu de reward hacking", fontweight="bold")
    axes[0].set_xlabel("Top-k analizat")
    axes[0].set_ylabel("Risk mediu")
    axes[0].grid(alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_title("Procent de molecule care trec auditul", fontweight="bold")
    axes[1].set_xlabel("Top-k analizat")
    axes[1].set_ylabel("Audit pass rate (%)")
    axes[1].grid(alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle("Comparatie intre scoring naiv si ranking protejat", fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "ml_04_multi_agent_ablation.png")


def build_reward_hacking_challenge(summary_df: pd.DataFrame) -> None:
    df = summary_df.copy()
    df = df[df["n"].fillna(0) > 0].copy()
    df["cohort_label"] = df["cohort"].str.replace("_", "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    colors = ["#2563EB", "#F97316", "#DC2626", "#7C3AED"]

    bars_shift = axes[0].bar(df["cohort_label"], df["mean_rank_shift"], color=colors[: len(df)], width=0.62)
    axes[0].set_title("Cat de mult sunt retrogradate cohortele problema", fontweight="bold")
    axes[0].set_ylabel("Mean rank shift")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].bar_label(bars_shift, labels=[f"{value:.0f}" for value in df["mean_rank_shift"]], padding=3)

    demote = (df["demoted_20plus_rate"] * 100.0).tolist()
    bars_demote = axes[1].bar(df["cohort_label"], demote, color=colors[: len(df)], width=0.62)
    axes[1].set_title("Procent retrogradat cu cel putin 20 pozitii", fontweight="bold")
    axes[1].set_ylabel("Demoted >=20 (%)")
    axes[1].set_ylim(0, max(demote) * 1.2 if demote else 100)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].bar_label(bars_demote, labels=[f"{value:.0f}%" for value in demote], padding=3)

    fig.suptitle("Testul de rezistenta la reward hacking", fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "ml_05_reward_hacking_challenge.png")


def build_optimization_trajectory(round_df: pd.DataFrame) -> None:
    df = round_df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"]).sort_values("round")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    axes[0].plot(df["round"], df["avg_score"], marker="o", linewidth=2.2, color="#2563EB", label="avg_score")
    axes[0].plot(df["round"], df["max_score"], marker="o", linewidth=2.2, color="#10B981", label="max_score")
    axes[0].set_title("Scorul candidatilor pe runde de optimizare", fontweight="bold")
    axes[0].set_xlabel("Runda")
    axes[0].set_ylabel("Scor")
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(df["round"], df["avg_pIC50"], marker="o", linewidth=2.2, color="#F59E0B", label="avg_pIC50")
    axes[1].plot(df["round"], df["avg_qed"], marker="o", linewidth=2.2, color="#7C3AED", label="avg_qed")
    axes[1].set_title("Potenza si calitatea chimica pe runde", fontweight="bold")
    axes[1].set_xlabel("Runda")
    axes[1].set_ylabel("Valoare medie")
    axes[1].legend(frameon=False, loc="best")
    axes[1].grid(alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Traiectoria optimizarii iterative", fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "ml_06_optimization_trajectory.png")


def build_candidate_tradeoff(batch_df: pd.DataFrame) -> None:
    df = batch_df.copy()
    for column in [
        "predicted_pIC50",
        "experimental_readiness_score",
        "cross_database_consensus_score",
        "prospective_batch_rank",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["predicted_pIC50", "experimental_readiness_score", "cross_database_consensus_score"])

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    scatter = ax.scatter(
        df["predicted_pIC50"],
        df["experimental_readiness_score"],
        c=df["cross_database_consensus_score"],
        cmap="viridis",
        s=90,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cross-database consensus")

    top_rows = df.nsmallest(6, "prospective_batch_rank")
    for _, row in top_rows.iterrows():
        ax.annotate(
            f"#{int(row['prospective_batch_rank'])}",
            (row["predicted_pIC50"], row["experimental_readiness_score"]),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=9,
            color="#111827",
        )

    ax.set_title("Compromisul dintre potenta si readiness experimental", fontweight="bold")
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Experimental readiness score")
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "ml_07_candidate_tradeoff.png")


def build_agent_support_heatmap(batch_df: pd.DataFrame) -> None:
    support_columns = [
        "potency_support",
        "chemistry_support",
        "safety_support",
        "domain_support",
        "structure_agent_support",
        "external_evidence_support",
        "evidence_arbiter_support",
    ]
    df = batch_df.copy()
    keep_cols = ["prospective_batch_rank", *support_columns]
    df = df[keep_cols].copy()
    for column in keep_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["prospective_batch_rank"]).sort_values("prospective_batch_rank").head(12)

    matrix = df[support_columns].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Support score")

    ax.set_title("Profilul multi-agent al candidatilor din lotul prospectiv", fontweight="bold")
    ax.set_xticks(np.arange(len(support_columns)))
    ax.set_xticklabels(
        [
            "potency",
            "chemistry",
            "safety",
            "domain",
            "structure",
            "external",
            "arbiter",
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([f"#{int(rank)}" for rank in df["prospective_batch_rank"]])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#0F172A" if value < 0.75 else "white",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "ml_08_agent_support_heatmap.png")


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_summary = _load_json(REPORTS_DIR / "model_performance_summary.json")
    source_holdout_df = pd.read_csv(REPORTS_DIR / "source_holdout_benchmark.csv")
    predictions_df = pd.read_csv(REPORTS_DIR / "source_holdout_predictions.csv")
    ablation_df = pd.read_csv(REPORTS_DIR / "multi_agent_ablation.csv")
    reward_hacking_df = pd.read_csv(REPORTS_DIR / "reward_hacking_challenge" / "reward_hacking_challenge_summary.csv")
    round_df = pd.read_csv(REPORTS_DIR / "optimization_round_summary.csv")
    batch_df = pd.read_csv(REPORTS_DIR / "prospective_validation_batch.csv")

    build_split_performance(model_summary)
    build_source_holdout_overview(source_holdout_df)
    build_uncertainty_vs_error(predictions_df)
    build_multi_agent_ablation(ablation_df)
    build_reward_hacking_challenge(reward_hacking_df)
    build_optimization_trajectory(round_df)
    build_candidate_tradeoff(batch_df)
    build_agent_support_heatmap(batch_df)

    print(f"[OK] Saved ML chapter figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
