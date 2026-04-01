from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA

from src.config import PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.pipelines.artifact_utils import load_csv_artifact


NOTEBOOK_DIR = PROJECT_ROOT / "reports" / "technical_notebook"
REQUIRED_RANKED_COLUMNS = [
    "smiles",
    "predicted_pIC50",
    "QED",
    "SA_score",
    "novelty_score",
    "applicability_score",
    "reward_hacking_risk",
    "naive_score",
    "final_score",
    "naive_rank",
    "rank",
    "audit_status",
    "audit_pass",
    "audit_demote_positions",
    "potency_support",
    "chemistry_support",
    "safety_support",
    "domain_support",
    "agent_disagreement_score",
    "multi_agent_balance",
]


def _optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _first_existing_csv(*paths: Path) -> pd.DataFrame | None:
    for path in paths:
        df = _optional_csv(path)
        if df is not None:
            return df
    return None


def _optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df.empty:
        return "_No rows available._"

    cols = [column for column in columns if column in df.columns]
    if not cols:
        return "_Columns unavailable._"

    subset = df[cols].head(n).copy()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, separator]

    for _, row in subset.iterrows():
        values = []
        for column in cols:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def _sample_frame(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    return df.sample(limit, random_state=42).reset_index(drop=True)


def _plot_pipeline_flowchart(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        {
            "xy": (0.03, 0.58),
            "label": "Public EGFR data\nChEMBL, BindingDB,\nPapyrus, ExCAPE, PubChem",
            "color": "#d8f3dc",
        },
        {
            "xy": (0.22, 0.58),
            "label": "Curation and features\nSMILES cleanup,\ndescriptors, ECFP",
            "color": "#cfe8ff",
        },
        {
            "xy": (0.41, 0.58),
            "label": "Multiview QSAR\nrandom, scaffold,\ntemporal validation",
            "color": "#dbe7c9",
        },
        {
            "xy": (0.60, 0.58),
            "label": "Candidate generation\nanalogs, iterative design,\nRL branches",
            "color": "#fdecc8",
        },
        {
            "xy": (0.79, 0.58),
            "label": "Protected ranking\nmulti-agent scoring,\naudit, veto, novelty",
            "color": "#ffd6cf",
        },
        {
            "xy": (0.31, 0.15),
            "label": "Orthogonal evidence\nstructural rescoring,\nexternal databases,\nfeasibility and readiness",
            "color": "#f1d1ff",
        },
        {
            "xy": (0.60, 0.15),
            "label": "Decision output\nprospective batch,\nbenchmarks, notebook",
            "color": "#ffe9a8",
        },
    ]

    width = 0.15
    height = 0.18
    for box in boxes:
        x, y = box["xy"]
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#264653",
            facecolor=box["color"],
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            box["label"],
            ha="center",
            va="center",
            fontsize=9,
            color="#1f2933",
        )

    arrows = [
        ((0.18, 0.67), (0.22, 0.67)),
        ((0.37, 0.67), (0.41, 0.67)),
        ((0.56, 0.67), (0.60, 0.67)),
        ((0.75, 0.67), (0.79, 0.67)),
        ((0.865, 0.58), (0.67, 0.33)),
        ((0.46, 0.58), (0.39, 0.33)),
        ((0.48, 0.24), (0.60, 0.24)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color="#264653",
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        0.5,
        0.93,
        "OncoForge pipeline from curated EGFR data to a prospective validation batch",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1f2933",
    )
    ax.text(
        0.5,
        0.04,
        "The main design idea is sequential narrowing: predict, generate, audit, validate, then hand off a smaller and safer shortlist.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4a5568",
    )
    _save_figure(fig, out_dir / "pipeline_flowchart.png")


def _plot_single_vs_multi_agent(ablation_df: pd.DataFrame | None, out_dir: Path) -> None:
    if ablation_df is None or ablation_df.empty:
        return

    strategy_map = {
        "naive_proxy": "Single-agent proxy",
        "protected_final": "Protected multi-agent",
    }
    plot_df = ablation_df[ablation_df["strategy"].isin(strategy_map)].copy()
    if plot_df.empty:
        return

    plot_df["strategy_label"] = plot_df["strategy"].map(strategy_map)
    plot_df = plot_df.sort_values(["strategy_label", "top_k"]).reset_index(drop=True)
    colors = {
        "Single-agent proxy": "#e76f51",
        "Protected multi-agent": "#2a9d8f",
    }
    metric_specs = [
        ("mean_predicted_pIC50", "Mean predicted pIC50"),
        ("mean_reward_hacking_risk", "Mean reward-hacking risk"),
        ("audit_pass_rate", "Audit pass rate"),
        ("review_or_fail_rate", "Review or fail rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        for strategy_label, color in colors.items():
            subset = plot_df[plot_df["strategy_label"] == strategy_label]
            if subset.empty or metric not in subset.columns:
                continue
            ax.plot(
                subset["top_k"],
                subset[metric],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=color,
                label=strategy_label,
            )
        ax.set_title(title)
        ax.grid(alpha=0.18)

    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_xlabel("Top-k shortlist size")
    axes[1, 1].set_xlabel("Top-k shortlist size")
    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle("Single-Agent Proxy Ranking vs Protected Multi-Agent Selection", fontsize=14)
    _save_figure(fig, out_dir / "single_agent_vs_multi_agent.png")


def _plot_risk_distribution(ranked: pd.DataFrame, out_dir: Path) -> None:
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"pass": "#2a9d8f", "review": "#e9c46a", "fail": "#e76f51"}

    for status in ["pass", "review", "fail"]:
        subset = ranked.loc[ranked["audit_status"] == status, "reward_hacking_risk"]
        if subset.empty:
            continue
        ax.hist(
            subset,
            bins=bins,
            alpha=0.70,
            label=f"{status.title()} ({len(subset)})",
            color=palette[status],
        )

    ax.set_xlabel("Reward hacking risk")
    ax.set_ylabel("Molecule count")
    ax.set_title("Audit Risk Distribution Across Multi-Agent Decisions")
    ax.legend(frameon=False)
    _save_figure(fig, out_dir / "risk_distribution_by_audit_status.png")


def _plot_naive_vs_verified(ranked: pd.DataFrame, out_dir: Path) -> None:
    sample = _sample_frame(ranked, 2500)
    demoted = ranked.sort_values(
        ["audit_demote_positions", "reward_hacking_risk"],
        ascending=[False, False],
    ).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        sample["naive_score"],
        sample["final_score"],
        c=sample["reward_hacking_risk"],
        cmap="viridis",
        alpha=0.65,
        s=18,
    )
    bounds = [
        min(sample["naive_score"].min(), sample["final_score"].min()),
        max(sample["naive_score"].max(), sample["final_score"].max()),
    ]
    ax.plot(bounds, bounds, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.scatter(
        demoted["naive_score"],
        demoted["final_score"],
        color="#d62828",
        edgecolor="black",
        linewidth=0.6,
        s=45,
        label="Most demoted by audit",
    )

    for _, row in demoted.head(6).iterrows():
        label = f"n{int(row['naive_rank'])}->f{int(row['rank'])}"
        ax.annotate(label, (row["naive_score"], row["final_score"]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Naive score")
    ax.set_ylabel("Protected final score")
    ax.set_title("Naive Reward vs Protected Multi-Agent Ranking")
    ax.legend(frameon=False, loc="lower right")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Reward hacking risk")
    _save_figure(fig, out_dir / "naive_vs_protected_scores.png")


def _plot_rank_shift(ranked: pd.DataFrame, out_dir: Path) -> None:
    demoted = ranked.loc[ranked["audit_demote_positions"] > 0].copy()
    demoted = demoted.sort_values(
        ["audit_demote_positions", "reward_hacking_risk"],
        ascending=[False, False],
    ).head(15)

    if demoted.empty:
        return

    labels = [f"Rank {int(row['rank'])}" for _, row in demoted.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, demoted["audit_demote_positions"], color="#bc4749")
    ax.invert_yaxis()
    ax.set_xlabel("Positions lost after anti-hacking audit")
    ax.set_ylabel("Protected rank")
    ax.set_title("Candidates Demoted by the Multi-Agent Audit")
    _save_figure(fig, out_dir / "audit_rank_demotions.png")

    demoted[
        [
            "rank",
            "naive_rank",
            "audit_demote_positions",
            "predicted_pIC50",
            "QED",
            "reward_hacking_risk",
            "audit_status",
            "smiles",
        ]
    ].to_csv(out_dir / "top_audit_demotions.csv", index=False)


def _plot_agent_support_heatmap(ranked: pd.DataFrame, out_dir: Path) -> None:
    support_columns = [
        "potency_support",
        "chemistry_support",
        "safety_support",
        "domain_support",
    ]
    lead_df = ranked.head(12).copy()
    matrix = lead_df[support_columns].to_numpy(dtype=float).T

    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(lead_df)))
    ax.set_xticklabels([f"#{rank}" for rank in lead_df["rank"]], rotation=0)
    ax.set_yticks(np.arange(len(support_columns)))
    ax.set_yticklabels(["Potency", "Chemistry", "Safety", "Domain"])
    ax.set_xlabel("Top protected candidates")
    ax.set_title("Multi-Agent Support Profile for Top Leads")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Support score")
    _save_figure(fig, out_dir / "top_leads_agent_support_heatmap.png")


def _plot_novelty_vs_applicability(ranked: pd.DataFrame, out_dir: Path) -> None:
    sample = _sample_frame(ranked, 2500)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        sample["novelty_score"],
        sample["applicability_score"],
        c=sample["reward_hacking_risk"],
        cmap="plasma",
        alpha=0.65,
        s=18,
    )
    ax.axvline(0.80, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.axhline(0.25, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Novelty score")
    ax.set_ylabel("Applicability score")
    ax.set_title("Novelty Must Stay Inside the Evidence Envelope")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Reward hacking risk")
    _save_figure(fig, out_dir / "novelty_vs_applicability.png")


def _plot_market_comparison(
    market_df: pd.DataFrame | None,
    generated_df: pd.DataFrame | None,
    shortlist_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    groups = []
    if market_df is not None and not market_df.empty:
        groups.append(("Marketed EGFR", market_df))
    if generated_df is not None and not generated_df.empty:
        groups.append(("Generated", generated_df))
    if shortlist_df is not None and not shortlist_df.empty:
        groups.append(("Novel shortlist", shortlist_df))

    if len(groups) < 2:
        return

    metrics = ["predicted_pIC50", "QED", "reward_hacking_risk"]
    titles = ["Predicted pIC50", "QED", "Reward hacking risk"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, metric, title in zip(axes, metrics, titles):
        metric_groups = [(label, frame) for label, frame in groups if metric in frame.columns]
        if len(metric_groups) < 2:
            ax.axis("off")
            ax.set_title(f"{title}\n(not enough artifacts)")
            continue

        data = [frame[metric].dropna().tolist() for _, frame in metric_groups]
        labels = [label for label, _ in metric_groups]
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle("Marketed Drugs vs Generated Candidate Quality")
    _save_figure(fig, out_dir / "marketed_vs_generated_boxplots.png")


def _plot_model_split_performance(model_summary: dict | None, out_dir: Path) -> None:
    if not model_summary:
        return

    split_labels = []
    rmses = []
    r2s = []
    for split_key, label in [
        ("random_split", "Random"),
        ("scaffold_split", "Scaffold"),
        ("temporal_split", "Temporal"),
    ]:
        split_metrics = model_summary.get(split_key)
        if not split_metrics:
            continue
        split_labels.append(label)
        rmses.append(float(split_metrics.get("rmse", 0.0)))
        r2s.append(float(split_metrics.get("r2", 0.0)))

    if not split_labels:
        return

    x = np.arange(len(split_labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x, rmses, color=["#457b9d", "#2a9d8f", "#e76f51"][: len(x)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_labels)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Model Error Across Validation Regimes")

    axes[1].bar(x, r2s, color=["#457b9d", "#2a9d8f", "#e76f51"][: len(x)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_labels)
    axes[1].set_ylabel("R2")
    axes[1].set_title("Generalization Strength Across Splits")
    fig.suptitle("Multiview Ensemble Validation Profile")
    _save_figure(fig, out_dir / "model_split_performance.png")


def _plot_uncertainty_calibration(model_summary: dict | None, out_dir: Path) -> None:
    if not model_summary:
        return
    calibration = model_summary.get("uncertainty_calibration") or {}
    split_entries = []
    for split_key, label in [
        ("random_split", "Random"),
        ("scaffold_split", "Scaffold"),
        ("temporal_split", "Temporal"),
    ]:
        metrics = calibration.get(split_key)
        if not metrics:
            continue
        split_entries.append(
            (
                label,
                float(metrics.get("raw_one_sigma_coverage", 0.0)),
                float(metrics.get("calibrated_one_sigma_coverage", 0.0)),
            )
        )
    if not split_entries:
        return

    labels = [item[0] for item in split_entries]
    raw_cov = [item[1] for item in split_entries]
    cal_cov = [item[2] for item in split_entries]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, raw_cov, width, label="Raw", color="#8ecae6")
    ax.bar(x + width / 2, cal_cov, width, label="Calibrated", color="#219ebc")
    ax.axhline(calibration.get("random_split", {}).get("target_one_sigma_coverage", 0.6827), linestyle="--", color="#6c757d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("One-sigma coverage")
    ax.set_title("Uncertainty Calibration Before and After Scaling")
    ax.legend(frameon=False)
    _save_figure(fig, out_dir / "uncertainty_calibration.png")


def _plot_gpu_gnn_benchmark(gpu_gnn_df: pd.DataFrame | None, out_dir: Path) -> None:
    if gpu_gnn_df is None or gpu_gnn_df.empty:
        return
    scaffold_df = gpu_gnn_df[gpu_gnn_df["split"] == "scaffold"].copy()
    if scaffold_df.empty:
        return
    order = ["multiview_reference", "gpu_graph_regressor", "consensus_blend"]
    scaffold_df["order_key"] = scaffold_df["model"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    scaffold_df = scaffold_df.sort_values(["order_key", "rmse"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(scaffold_df["model"], scaffold_df["rmse"], color=["#457b9d", "#e76f51", "#2a9d8f"][: len(scaffold_df)])
    ax.set_ylabel("Scaffold RMSE")
    ax.set_title("GPU Graph Model vs Classical Ensemble")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, scaffold_df["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _save_figure(fig, out_dir / "gpu_gnn_scaffold_benchmark.png")


def _plot_structural_rescoring(structural_df: pd.DataFrame | None, out_dir: Path) -> None:
    if structural_df is None or structural_df.empty or "docking_rescore" not in structural_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        structural_df["docking_rescore"],
        structural_df["final_score"],
        c=structural_df.get("predicted_pIC50", pd.Series(np.zeros(len(structural_df)))),
        cmap="viridis",
        alpha=0.70,
        s=24,
    )
    ax.axvline(0.45, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Structural rescoring support")
    ax.set_ylabel("Final score")
    ax.set_title("Orthogonal Structural Rescoring of Top Optimized Candidates")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Predicted pIC50")
    _save_figure(fig, out_dir / "structural_rescoring_scatter.png")


def _plot_vina_affinity(structural_df: pd.DataFrame | None, out_dir: Path) -> None:
    if structural_df is None or structural_df.empty or "vina_affinity_kcal" not in structural_df.columns:
        return

    vina_df = structural_df.dropna(subset=["vina_affinity_kcal"]).copy()
    if vina_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        vina_df["vina_affinity_kcal"],
        vina_df["structural_priority_score"] if "structural_priority_score" in vina_df.columns else vina_df["final_score"],
        c=vina_df.get("feasibility_score", vina_df.get("predicted_pIC50", pd.Series(np.zeros(len(vina_df))))),
        cmap="viridis",
        alpha=0.72,
        s=24,
    )
    ax.axvline(-8.0, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("AutoDock Vina affinity (kcal/mol)")
    ax.set_ylabel("Structural priority score")
    ax.set_title("Docking Strength vs Final Candidate Priority")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Predicted pIC50")
    _save_figure(fig, out_dir / "vina_affinity_vs_priority.png")


def _plot_interaction_support(structural_df: pd.DataFrame | None, out_dir: Path) -> None:
    if structural_df is None or structural_df.empty or "interaction_support_score" not in structural_df.columns:
        return

    interaction_df = structural_df.dropna(subset=["interaction_support_score"]).copy()
    if interaction_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        interaction_df.get("vina_affinity_kcal", pd.Series(np.zeros(len(interaction_df)))),
        interaction_df["interaction_support_score"],
        c=interaction_df.get("predicted_pIC50", pd.Series(np.zeros(len(interaction_df)))),
        cmap="viridis",
        alpha=0.72,
        s=24,
    )
    ax.axhline(0.45, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("AutoDock Vina affinity (kcal/mol)")
    ax.set_ylabel("Interaction support score")
    ax.set_title("Residue-Level Interaction Support for Docked Leads")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Predicted pIC50")
    _save_figure(fig, out_dir / "interaction_support_vs_vina.png")


def _plot_readiness_vs_structure(readiness_df: pd.DataFrame | None, out_dir: Path) -> None:
    if readiness_df is None or readiness_df.empty or "experimental_readiness_score" not in readiness_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        readiness_df.get("docking_rescore", pd.Series(np.zeros(len(readiness_df)))),
        readiness_df["experimental_readiness_score"],
        c=readiness_df.get("interaction_support_score", readiness_df.get("feasibility_score", pd.Series(np.zeros(len(readiness_df))))),
        cmap="viridis",
        alpha=0.72,
        s=24,
    )
    ax.axhline(0.70, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.axvline(0.50, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Docking rescore")
    ax.set_ylabel("Experimental readiness score")
    ax.set_title("Experimental Readiness vs Structural Support")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Interaction / feasibility support")
    _save_figure(fig, out_dir / "readiness_vs_structure.png")


def _plot_cross_database_vs_potency(crossdb_df: pd.DataFrame | None, out_dir: Path) -> None:
    if crossdb_df is None or crossdb_df.empty or "cross_database_consensus_score" not in crossdb_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        crossdb_df["cross_database_consensus_score"],
        crossdb_df["predicted_pIC50"],
        c=crossdb_df.get("experimental_readiness_score", crossdb_df.get("feasibility_score", pd.Series(np.zeros(len(crossdb_df))))),
        cmap="viridis",
        alpha=0.72,
        s=24,
    )
    ax.axvline(0.55, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("Cross-database consensus score")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("Independent Database Support for Top Candidates")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Readiness / feasibility support")
    _save_figure(fig, out_dir / "cross_database_vs_potency.png")


def _plot_pubchem_assay_relevance(assay_catalog_df: pd.DataFrame | None, out_dir: Path) -> None:
    if assay_catalog_df is None or assay_catalog_df.empty or "mean_record_relevance" not in assay_catalog_df.columns:
        return
    plot_df = assay_catalog_df.sort_values(
        ["mean_record_relevance", "n_records"],
        ascending=[False, False],
    ).head(12).copy()
    if plot_df.empty:
        return
    colors = plot_df.get("assay_support_tier", pd.Series("moderate", index=plot_df.index)).map(
        {"strong": "#2a9d8f", "moderate": "#e9c46a", "weak": "#e76f51"}
    ).fillna("#6c757d")
    labels = plot_df["assay_name"].fillna("unknown").astype(str).str.slice(0, 52)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, plot_df["mean_record_relevance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mean assay relevance")
    ax.set_title("PubChem EGFR Assays Ranked by Relevance")
    _save_figure(fig, out_dir / "pubchem_assay_relevance.png")


def _plot_structural_benchmark_boxplots(
    market_df: pd.DataFrame | None,
    structural_df: pd.DataFrame | None,
    generated_df: pd.DataFrame | None,
    prospective_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    groups = []
    if market_df is not None and not market_df.empty:
        groups.append(("Marketed", market_df))
    if structural_df is not None and not structural_df.empty:
        groups.append(("Optimized", structural_df))
    if generated_df is not None and not generated_df.empty:
        groups.append(("Generated", generated_df))
    if prospective_df is not None and not prospective_df.empty:
        groups.append(("Prospective", prospective_df))
    if len(groups) < 2:
        return

    metrics = [("vina_affinity_kcal", "Vina affinity"), ("interaction_support_score", "Interaction support")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        metric_groups = [(label, frame) for label, frame in groups if metric in frame.columns and not frame[metric].dropna().empty]
        if len(metric_groups) < 2:
            ax.axis("off")
            continue
        data = [frame[metric].dropna().tolist() for _, frame in metric_groups]
        labels = [label for label, _ in metric_groups]
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Structural Benchmark: Marketed vs Generated Molecules")
    _save_figure(fig, out_dir / "structural_benchmark_boxplots.png")


def _plot_generator_benchmark_overview(out_dir: Path) -> None:
    summary_specs = [
        ("Broad analogs", _optional_json(PROJECT_ROOT / "reports" / "generated_analogs_ranked.summary.json")),
        ("AI-guided", _optional_json(PROJECT_ROOT / "reports" / "ai_guided_analogs.summary.json")),
        ("Iterative", _optional_json(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.summary.json")),
    ]
    rows = [(label, summary) for label, summary in summary_specs if summary]
    if len(rows) < 2:
        return

    labels = [label for label, _ in rows]
    candidate_counts = [float(summary.get("n_candidates", 0.0)) for _, summary in rows]
    top_scores = [float(summary.get("top_mean_final_score", 0.0)) for _, summary in rows]
    generator_scores = [float(summary.get("mean_generator_priority_score", 0.0)) for _, summary in rows]
    audit_pass = [float(summary.get("audit_pass_rate", 0.0)) for _, summary in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plots = [
        ("Candidate count", candidate_counts, "#457b9d"),
        ("Top mean final score", top_scores, "#2a9d8f"),
        ("Mean generator priority", generator_scores, "#e9c46a"),
        ("Audit pass rate", audit_pass, "#e76f51"),
    ]
    for ax, (title, values, color) in zip(axes.flatten(), plots):
        ax.bar(labels, values, color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Generator Upgrade Benchmark Across Candidate Pipelines")
    _save_figure(fig, out_dir / "generator_benchmark_overview.png")


def _copy_plot_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.write_bytes(source.read_bytes())


def _featurize_smiles(smiles_list: list[str]) -> tuple[np.ndarray, list[str]]:
    features = []
    valid_smiles = []

    for smiles in smiles_list:
        try:
            fp = ecfp_from_smiles(smiles)
        except Exception:
            continue
        if fp is None:
            continue
        features.append(fp)
        valid_smiles.append(smiles)

    if not features:
        return np.empty((0, 2048), dtype=float), []

    return np.vstack(features), valid_smiles


def _plot_chemical_space_snapshot(
    ranked: pd.DataFrame,
    market_df: pd.DataFrame | None,
    shortlist_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    ranked_sample = ranked.head(700)
    market_sample = market_df if market_df is not None else pd.DataFrame()
    shortlist_sample = shortlist_df if shortlist_df is not None else pd.DataFrame()

    groups: list[tuple[str, pd.DataFrame, str, float]] = [
        ("Ranked leads", ranked_sample, "#1d3557", 0.30),
    ]
    if not market_sample.empty:
        groups.append(("Marketed EGFR", market_sample, "#e76f51", 0.90))
    if not shortlist_sample.empty:
        groups.append(("Novel shortlist", shortlist_sample.head(120), "#2a9d8f", 0.90))

    all_features = []
    group_sizes = []
    plotted_groups: list[tuple[str, str, float]] = []

    for label, frame, color, alpha in groups:
        smiles_column = "smiles"
        smiles = frame[smiles_column].dropna().tolist()
        features, valid_smiles = _featurize_smiles(smiles)
        if features.size == 0 or not valid_smiles:
            continue
        all_features.append(features)
        group_sizes.append(len(valid_smiles))
        plotted_groups.append((label, color, alpha))

    if len(all_features) < 2:
        return

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(np.vstack(all_features))

    fig, ax = plt.subplots(figsize=(8, 6))
    start = 0
    for (label, color, alpha), size in zip(plotted_groups, group_sizes):
        end = start + size
        coords = transformed[start:end]
        ax.scatter(coords[:, 0], coords[:, 1], label=label, alpha=alpha, s=18, color=color)
        start = end

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Chemical Space Snapshot for the Technical Notebook")
    ax.legend(frameon=False)
    _save_figure(fig, out_dir / "technical_notebook_chemical_space.png")


def _write_summary(
    ranked: pd.DataFrame,
    market_df: pd.DataFrame | None,
    generated_df: pd.DataFrame | None,
    shortlist_df: pd.DataFrame | None,
    model_summary: dict | None,
    structural_df: pd.DataFrame | None,
    feasibility_df: pd.DataFrame | None,
    readiness_df: pd.DataFrame | None,
    crossdb_df: pd.DataFrame | None,
    prospective_df: pd.DataFrame | None,
    rl_df: pd.DataFrame | None,
    rl_summary: dict | None,
    out_dir: Path,
) -> None:
    demoted = ranked.sort_values(
        ["audit_demote_positions", "reward_hacking_risk"],
        ascending=[False, False],
    )
    promoted = ranked.sort_values(
        ["audit_promote_positions", "final_score"],
        ascending=[False, False],
    )

    status_counts = ranked["audit_status"].value_counts().to_dict()
    summary = {
        "ranked_molecules": int(len(ranked)),
        "audit_pass_rate": float((ranked["audit_status"] == "pass").mean()),
        "audit_review_rate": float((ranked["audit_status"] == "review").mean()),
        "audit_fail_rate": float((ranked["audit_status"] == "fail").mean()),
        "median_reward_hacking_risk": float(ranked["reward_hacking_risk"].median()),
        "median_agent_disagreement": float(ranked["agent_disagreement_score"].median()),
        "mean_audit_demotion": float(ranked["audit_demote_positions"].mean()),
        "status_counts": status_counts,
    }
    if model_summary:
        summary["model_random_rmse"] = float(model_summary.get("random_split", {}).get("rmse", 0.0))
        summary["model_scaffold_rmse"] = float(model_summary.get("scaffold_split", {}).get("rmse", 0.0))
        if model_summary.get("temporal_split"):
            summary["model_temporal_rmse"] = float(model_summary.get("temporal_split", {}).get("rmse", 0.0))
    if structural_df is not None and not structural_df.empty and "docking_rescore" in structural_df.columns:
        summary["mean_structural_rescore"] = float(structural_df["docking_rescore"].mean())
    if structural_df is not None and not structural_df.empty and "vina_affinity_kcal" in structural_df.columns:
        valid_affinities = structural_df["vina_affinity_kcal"].dropna()
        if not valid_affinities.empty:
            summary["mean_vina_affinity_kcal"] = float(valid_affinities.mean())
            summary["best_vina_affinity_kcal"] = float(valid_affinities.min())
            summary["vina_docked_candidates"] = int(valid_affinities.shape[0])
    if structural_df is not None and not structural_df.empty and "interaction_support_score" in structural_df.columns:
        summary["mean_interaction_support"] = float(structural_df["interaction_support_score"].mean())
        summary["best_interaction_support"] = float(structural_df["interaction_support_score"].max())
    if feasibility_df is not None and not feasibility_df.empty:
        summary["feasibility_pass_rate"] = float((feasibility_df["feasibility_status"] == "pass").mean())
        summary["mean_feasibility_score"] = float(feasibility_df["feasibility_score"].mean())
    if readiness_df is not None and not readiness_df.empty and "experimental_readiness_score" in readiness_df.columns:
        summary["mean_experimental_readiness"] = float(readiness_df["experimental_readiness_score"].mean())
        summary["readiness_ready_rate"] = float((readiness_df["experimental_readiness_status"] == "ready").mean())
        if "evidence_arbiter_support" in readiness_df.columns:
            summary["evidence_arbiter_mean_support"] = float(readiness_df["evidence_arbiter_support"].mean())
        if "evidence_arbiter_status" in readiness_df.columns:
            summary["evidence_arbiter_pass_rate"] = float((readiness_df["evidence_arbiter_status"] == "pass").mean())
    if crossdb_df is not None and not crossdb_df.empty:
        summary["cross_database_mean_consensus"] = float(crossdb_df["cross_database_consensus_score"].mean())
        summary["cross_database_strong_rate"] = float((crossdb_df["cross_database_status"] == "strong").mean())
        summary["cross_database_moderate_rate"] = float((crossdb_df["cross_database_status"] == "moderate").mean())
        if "external_evidence_support" in crossdb_df.columns:
            summary["external_evidence_mean_support"] = float(crossdb_df["external_evidence_support"].mean())
        if "external_evidence_status" in crossdb_df.columns:
            summary["external_evidence_pass_rate"] = float((crossdb_df["external_evidence_status"] == "pass").mean())
    if prospective_df is not None and not prospective_df.empty:
        summary["prospective_batch_size"] = int(len(prospective_df))
        summary["prospective_mean_acquisition_score"] = float(prospective_df["prospective_acquisition_score"].mean())
        summary["prospective_mean_readiness"] = float(prospective_df["experimental_readiness_score"].mean())
        if "cross_database_consensus_score" in prospective_df.columns:
            summary["prospective_mean_cross_database_consensus"] = float(prospective_df["cross_database_consensus_score"].mean())
        if "external_evidence_support" in prospective_df.columns:
            summary["prospective_mean_external_evidence"] = float(prospective_df["external_evidence_support"].mean())
        if "structure_evidence_support" in prospective_df.columns:
            summary["prospective_mean_structure_evidence_support"] = float(prospective_df["structure_evidence_support"].mean())
        if "structure_evidence_pareto_is_front" in prospective_df.columns:
            summary["prospective_pareto_front_rate"] = float(prospective_df["structure_evidence_pareto_is_front"].mean())
    if rl_df is not None and not rl_df.empty:
        summary["rl_top_mean_feasibility"] = float(rl_df["feasibility_score"].head(20).mean())
        summary["rl_top_mean_pic50"] = float(rl_df["predicted_pIC50"].head(20).mean())
        if "cross_database_consensus_score" in rl_df.columns:
            summary["rl_mean_cross_database_consensus"] = float(rl_df["cross_database_consensus_score"].head(20).mean())
        if "external_evidence_support" in rl_df.columns:
            summary["rl_mean_external_evidence_support"] = float(rl_df["external_evidence_support"].head(20).mean())
        if "structure_evidence_support" in rl_df.columns:
            summary["rl_mean_structure_evidence_support"] = float(rl_df["structure_evidence_support"].head(20).mean())
        if "experimental_readiness_status" in rl_df.columns:
            summary["rl_readiness_ready_rate"] = float((rl_df["experimental_readiness_status"].head(20) == "ready").mean())
    if rl_summary:
        summary["rl_best_episode_return"] = float(rl_summary.get("best_episode_return", 0.0))
    generator_suite = _optional_csv(PROJECT_ROOT / "reports" / "generation_benchmark_suite.csv")
    if generator_suite is not None and not generator_suite.empty:
        suite_map = {
            "generated_analogs_ranked": "generated",
            "ai_guided_analogs": "ai_guided",
            "iterative_ai_optimized_candidates": "iterative",
        }
        for benchmark_name, label in suite_map.items():
            rows = generator_suite[generator_suite["benchmark_name"] == benchmark_name]
            if rows.empty:
                continue
            row = rows.iloc[0]
            summary[f"{label}_candidate_count"] = int(row.get("n_candidates", 0))
            summary[f"{label}_mean_generator_priority"] = float(row.get("mean_generator_priority_score", 0.0))
            summary[f"{label}_mean_adaptive_action_prior"] = float(row.get("mean_adaptive_action_prior", 0.0))
            summary[f"{label}_top_mean_final_score"] = float(row.get("top_mean_final_score", 0.0))
            summary[f"{label}_audit_pass_rate"] = float(row.get("audit_pass_rate", 0.0))
            summary[f"{label}_cross_database_pass_rate"] = float(row.get("cross_database_pass_rate", 0.0))
            summary[f"{label}_external_evidence_pass_rate"] = float(row.get("external_evidence_pass_rate", 0.0))
            summary[f"{label}_parent_improvement_rate_final_score"] = float(row.get("parent_improvement_rate_final_score", 0.0))
            summary[f"{label}_internal_diversity"] = float(row.get("internal_diversity", 0.0))
            summary[f"{label}_strong_transformation_memory_rate"] = float(row.get("strong_transformation_memory_rate", 0.0))
    else:
        generator_summaries = {
            "generated": _optional_json(PROJECT_ROOT / "reports" / "generated_analogs_ranked.summary.json"),
            "ai_guided": _optional_json(PROJECT_ROOT / "reports" / "ai_guided_analogs.summary.json"),
            "iterative": _optional_json(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.summary.json"),
        }
        for label, generator_summary in generator_summaries.items():
            if not generator_summary:
                continue
            summary[f"{label}_candidate_count"] = int(generator_summary.get("n_candidates", 0))
            summary[f"{label}_mean_generator_priority"] = float(generator_summary.get("mean_generator_priority_score", 0.0))
            summary[f"{label}_mean_adaptive_action_prior"] = float(generator_summary.get("mean_adaptive_action_prior", 0.0))
            summary[f"{label}_top_mean_final_score"] = float(generator_summary.get("top_mean_final_score", 0.0))
            summary[f"{label}_audit_pass_rate"] = float(generator_summary.get("audit_pass_rate", 0.0))
    gpu_gnn_summary = _optional_json(PROJECT_ROOT / "reports" / "gpu_gnn_performance_summary.json")
    if gpu_gnn_summary:
        scaffold_rows = [row for row in gpu_gnn_summary.get("splits", []) if row.get("split") == "scaffold"]
        if scaffold_rows:
            best_row = sorted(scaffold_rows, key=lambda row: float(row.get("rmse", 999.0)))[0]
            summary["gpu_gnn_best_scaffold_model"] = str(best_row.get("model", "n/a"))
            summary["gpu_gnn_best_scaffold_rmse"] = float(best_row.get("rmse", 0.0))
    gpu_rl_df = _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_dqn" / "gpu_rl_top_candidates.csv")
    gpu_rl_summary = _optional_json(PROJECT_ROOT / "reports" / "rl_gpu_dqn" / "gpu_rl_training_summary.json")
    if gpu_rl_df is not None and not gpu_rl_df.empty:
        summary["gpu_rl_mean_pic50"] = float(gpu_rl_df["predicted_pIC50"].head(20).mean())
        if "cross_database_consensus_score" in gpu_rl_df.columns:
            summary["gpu_rl_mean_cross_database_consensus"] = float(gpu_rl_df["cross_database_consensus_score"].head(20).mean())
        if "external_evidence_support" in gpu_rl_df.columns:
            summary["gpu_rl_mean_external_evidence_support"] = float(gpu_rl_df["external_evidence_support"].head(20).mean())
        if "evidence_arbiter_support" in gpu_rl_df.columns:
            summary["gpu_rl_mean_evidence_arbiter_support"] = float(gpu_rl_df["evidence_arbiter_support"].head(20).mean())
        if "structure_evidence_support" in gpu_rl_df.columns:
            summary["gpu_rl_mean_structure_evidence_support"] = float(gpu_rl_df["structure_evidence_support"].head(20).mean())
    if gpu_rl_summary:
        summary["gpu_rl_best_episode_return"] = float(gpu_rl_summary.get("best_episode_return", 0.0))
    actor_critic_df = _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_actor_critic" / "gpu_actor_critic_top_candidates.csv")
    actor_critic_summary = _optional_json(PROJECT_ROOT / "reports" / "rl_gpu_actor_critic" / "gpu_actor_critic_summary.json")
    if actor_critic_df is not None and not actor_critic_df.empty:
        summary["gpu_actor_critic_mean_pic50"] = float(actor_critic_df["predicted_pIC50"].head(20).mean())
        if "cross_database_consensus_score" in actor_critic_df.columns:
            summary["gpu_actor_critic_mean_cross_database_consensus"] = float(actor_critic_df["cross_database_consensus_score"].head(20).mean())
        if "external_evidence_support" in actor_critic_df.columns:
            summary["gpu_actor_critic_mean_external_evidence_support"] = float(actor_critic_df["external_evidence_support"].head(20).mean())
        if "experimental_readiness_status" in actor_critic_df.columns:
            summary["gpu_actor_critic_ready_rate"] = float((actor_critic_df["experimental_readiness_status"].head(20) == "ready").mean())
        if "structure_evidence_support" in actor_critic_df.columns:
            summary["gpu_actor_critic_mean_structure_evidence_support"] = float(actor_critic_df["structure_evidence_support"].head(20).mean())
    if actor_critic_summary:
        summary["gpu_actor_critic_best_episode_return"] = float(actor_critic_summary.get("best_episode_return", 0.0))
    pubchem_summary = _optional_json(PROJECT_ROOT / "data" / "processed" / "pubchem_egfr_reference.summary.json")
    if pubchem_summary:
        summary["pubchem_mean_enriched_evidence_score"] = float(pubchem_summary.get("mean_enriched_evidence_score", 0.0))
        summary["pubchem_strong_evidence_rate"] = float(pubchem_summary.get("strong_evidence_rate", 0.0))
        summary["pubchem_virtual_proxy_exposed_rate"] = float(pubchem_summary.get("virtual_proxy_exposed_rate", 0.0))
    papyrus_summary = _optional_json(PROJECT_ROOT / "data" / "processed" / "papyrus_egfr_reference.summary.json")
    if papyrus_summary:
        summary["papyrus_unique_molecules"] = int(papyrus_summary.get("n_unique_molecules", 0))
        summary["papyrus_mean_support_score"] = float(papyrus_summary.get("mean_support_score", 0.0))
    excape_summary = _optional_json(PROJECT_ROOT / "data" / "processed" / "excape_egfr_reference.summary.json")
    if excape_summary:
        summary["excape_unique_molecules"] = int(excape_summary.get("n_unique_molecules", 0))
        summary["excape_mean_support_score"] = float(excape_summary.get("mean_support_score", 0.0))
    robustness_summary = _optional_csv(PROJECT_ROOT / "reports" / "model_robustness_summary.csv")
    if robustness_summary is not None and not robustness_summary.empty:
        scaffold_rows = robustness_summary[robustness_summary["split"] == "scaffold"].sort_values("robustness_score")
        if not scaffold_rows.empty:
            best_row = scaffold_rows.iloc[0]
            summary["best_robust_model_family"] = str(best_row["model_family"])
            summary["best_robust_scaffold_rmse"] = float(best_row["mean_rmse"])
            summary["best_robust_scaffold_rmse_std"] = float(best_row["std_rmse"])
    challenge_summary = _optional_csv(PROJECT_ROOT / "reports" / "reward_hacking_challenge" / "reward_hacking_challenge_summary.csv")
    if challenge_summary is not None and not challenge_summary.empty:
        trusted = challenge_summary[challenge_summary["cohort"] == "trusted_controls"]
        exploits = challenge_summary[challenge_summary["cohort"] == "proxy_exploits"]
        if not trusted.empty:
            summary["challenge_trusted_pass_rate"] = float(trusted.iloc[0]["audit_pass_rate"])
        if not exploits.empty:
            summary["challenge_proxy_demoted_rate"] = float(exploits.iloc[0]["demoted_20plus_rate"])
            summary["challenge_proxy_review_or_fail_rate"] = float(exploits.iloc[0]["review_or_fail_rate"])
    source_holdout_summary = _optional_csv(PROJECT_ROOT / "reports" / "source_holdout_benchmark.csv")
    if source_holdout_summary is not None and not source_holdout_summary.empty:
        summary["source_holdout_mean_rmse"] = float(source_holdout_summary["rmse"].mean())
        summary["source_holdout_best_source"] = str(source_holdout_summary.sort_values("rmse").iloc[0]["source"])
        summary["source_holdout_best_rmse"] = float(source_holdout_summary["rmse"].min())
        summary["source_holdout_mean_recall_top20pct"] = float(source_holdout_summary["recall_top20pct"].mean())
    rediscovery_summary = _optional_json(PROJECT_ROOT / "reports" / "rediscovery_benchmark" / "rediscovery_summary.json")
    if rediscovery_summary:
        summary["rediscovery_protected_top10_recall"] = float(rediscovery_summary.get("protected_top10_recall", 0.0))
        summary["rediscovery_naive_top10_recall"] = float(rediscovery_summary.get("naive_top10_recall", 0.0))
        summary["rediscovery_protected_top20_recall"] = float(rediscovery_summary.get("protected_top20_recall", 0.0))
        summary["rediscovery_naive_top20_recall"] = float(rediscovery_summary.get("naive_top20_recall", 0.0))
        summary["rediscovery_protected_median_positive_rank"] = float(rediscovery_summary.get("protected_median_positive_rank", 0.0))
        summary["rediscovery_naive_median_positive_rank"] = float(rediscovery_summary.get("naive_median_positive_rank", 0.0))

    (out_dir / "technical_notebook_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Technical Notebook Assets",
        "",
        "## Audit Overview",
        f"- Ranked molecules: `{len(ranked)}`",
        f"- Audit pass rate: `{summary['audit_pass_rate']:.3f}`",
        f"- Audit review rate: `{summary['audit_review_rate']:.3f}`",
        f"- Audit fail rate: `{summary['audit_fail_rate']:.3f}`",
        f"- Median reward hacking risk: `{summary['median_reward_hacking_risk']:.3f}`",
        f"- Median agent disagreement: `{summary['median_agent_disagreement']:.3f}`",
        "",
        "## Model Validation Snapshot",
        f"- Random RMSE: `{summary.get('model_random_rmse', float('nan')):.3f}`",
        f"- Scaffold RMSE: `{summary.get('model_scaffold_rmse', float('nan')):.3f}`",
        f"- Temporal RMSE: `{summary.get('model_temporal_rmse', float('nan')):.3f}`",
        "",
        "## Feasibility Snapshot",
        f"- Feasibility pass rate: `{summary.get('feasibility_pass_rate', float('nan')):.3f}`",
        f"- Mean feasibility score: `{summary.get('mean_feasibility_score', float('nan')):.3f}`",
        f"- Mean experimental readiness: `{summary.get('mean_experimental_readiness', float('nan')):.3f}`",
        f"- Experimental readiness ready rate: `{summary.get('readiness_ready_rate', float('nan')):.3f}`",
        f"- Evidence arbiter mean support: `{summary.get('evidence_arbiter_mean_support', float('nan')):.3f}`",
        f"- Evidence arbiter pass rate: `{summary.get('evidence_arbiter_pass_rate', float('nan')):.3f}`",
        f"- Cross-database mean consensus: `{summary.get('cross_database_mean_consensus', float('nan')):.3f}`",
        f"- Cross-database strong rate: `{summary.get('cross_database_strong_rate', float('nan')):.3f}`",
        f"- External evidence mean support: `{summary.get('external_evidence_mean_support', float('nan')):.3f}`",
        f"- External evidence pass rate: `{summary.get('external_evidence_pass_rate', float('nan')):.3f}`",
        f"- Papyrus molecules / mean support: `{summary.get('papyrus_unique_molecules', 0)}` / `{summary.get('papyrus_mean_support_score', float('nan')):.3f}`",
        f"- ExCAPE molecules / mean support: `{summary.get('excape_unique_molecules', 0)}` / `{summary.get('excape_mean_support_score', float('nan')):.3f}`",
        f"- PubChem mean enriched evidence: `{summary.get('pubchem_mean_enriched_evidence_score', float('nan')):.3f}`",
        f"- PubChem strong evidence rate: `{summary.get('pubchem_strong_evidence_rate', float('nan')):.3f}`",
        f"- PubChem virtual/proxy exposure rate: `{summary.get('pubchem_virtual_proxy_exposed_rate', float('nan')):.3f}`",
        f"- Mean Vina affinity: `{summary.get('mean_vina_affinity_kcal', float('nan')):.3f}` kcal/mol",
        f"- Best Vina affinity: `{summary.get('best_vina_affinity_kcal', float('nan')):.3f}` kcal/mol",
        f"- Mean interaction support: `{summary.get('mean_interaction_support', float('nan')):.3f}`",
        f"- Best interaction support: `{summary.get('best_interaction_support', float('nan')):.3f}`",
        f"- Prospective validation batch size: `{summary.get('prospective_batch_size', 0)}`",
        f"- Prospective mean acquisition score: `{summary.get('prospective_mean_acquisition_score', float('nan')):.3f}`",
        f"- Prospective mean cross-database consensus: `{summary.get('prospective_mean_cross_database_consensus', float('nan')):.3f}`",
        f"- Prospective mean external evidence: `{summary.get('prospective_mean_external_evidence', float('nan')):.3f}`",
        f"- Prospective mean structure-evidence support: `{summary.get('prospective_mean_structure_evidence_support', float('nan')):.3f}`",
        f"- Prospective Pareto-front rate: `{summary.get('prospective_pareto_front_rate', float('nan')):.3f}`",
        f"- Broad analog count / mean generator priority: `{summary.get('generated_candidate_count', 0)}` / `{summary.get('generated_mean_generator_priority', float('nan')):.3f}`",
        f"- Broad analog mean adaptive prior: `{summary.get('generated_mean_adaptive_action_prior', float('nan')):.3f}`",
        f"- Broad analog cross-db pass / parent improvement: `{summary.get('generated_cross_database_pass_rate', float('nan')):.3f}` / `{summary.get('generated_parent_improvement_rate_final_score', float('nan')):.3f}`",
        f"- AI-guided count / mean generator priority: `{summary.get('ai_guided_candidate_count', 0)}` / `{summary.get('ai_guided_mean_generator_priority', float('nan')):.3f}`",
        f"- AI-guided mean adaptive prior: `{summary.get('ai_guided_mean_adaptive_action_prior', float('nan')):.3f}`",
        f"- AI-guided cross-db pass / parent improvement: `{summary.get('ai_guided_cross_database_pass_rate', float('nan')):.3f}` / `{summary.get('ai_guided_parent_improvement_rate_final_score', float('nan')):.3f}`",
        f"- Iterative count / mean generator priority: `{summary.get('iterative_candidate_count', 0)}` / `{summary.get('iterative_mean_generator_priority', float('nan')):.3f}`",
        f"- Iterative top mean final score: `{summary.get('iterative_top_mean_final_score', float('nan')):.3f}`",
        f"- Iterative mean adaptive prior: `{summary.get('iterative_mean_adaptive_action_prior', float('nan')):.3f}`",
        f"- Iterative cross-db pass / parent improvement: `{summary.get('iterative_cross_database_pass_rate', float('nan')):.3f}` / `{summary.get('iterative_parent_improvement_rate_final_score', float('nan')):.3f}`",
        f"- RL top mean feasibility: `{summary.get('rl_top_mean_feasibility', float('nan')):.3f}`",
        f"- RL mean cross-database consensus: `{summary.get('rl_mean_cross_database_consensus', float('nan')):.3f}`",
        f"- RL mean external evidence support: `{summary.get('rl_mean_external_evidence_support', float('nan')):.3f}`",
        f"- RL mean structure-evidence support: `{summary.get('rl_mean_structure_evidence_support', float('nan')):.3f}`",
        f"- RL ready rate: `{summary.get('rl_readiness_ready_rate', float('nan')):.3f}`",
        f"- RL best episode return: `{summary.get('rl_best_episode_return', float('nan')):.3f}`",
        f"- GPU GNN best scaffold model: `{summary.get('gpu_gnn_best_scaffold_model', 'n/a')}`",
        f"- GPU GNN best scaffold RMSE: `{summary.get('gpu_gnn_best_scaffold_rmse', float('nan')):.3f}`",
        f"- GPU RL mean cross-database consensus: `{summary.get('gpu_rl_mean_cross_database_consensus', float('nan')):.3f}`",
        f"- GPU RL mean external evidence support: `{summary.get('gpu_rl_mean_external_evidence_support', float('nan')):.3f}`",
        f"- GPU RL mean evidence arbiter support: `{summary.get('gpu_rl_mean_evidence_arbiter_support', float('nan')):.3f}`",
        f"- GPU RL mean structure-evidence support: `{summary.get('gpu_rl_mean_structure_evidence_support', float('nan')):.3f}`",
        f"- GPU RL best episode return: `{summary.get('gpu_rl_best_episode_return', float('nan')):.3f}`",
        f"- GPU actor-critic mean cross-database consensus: `{summary.get('gpu_actor_critic_mean_cross_database_consensus', float('nan')):.3f}`",
        f"- GPU actor-critic mean external evidence support: `{summary.get('gpu_actor_critic_mean_external_evidence_support', float('nan')):.3f}`",
        f"- GPU actor-critic mean structure-evidence support: `{summary.get('gpu_actor_critic_mean_structure_evidence_support', float('nan')):.3f}`",
        f"- GPU actor-critic ready rate: `{summary.get('gpu_actor_critic_ready_rate', float('nan')):.3f}`",
        f"- GPU actor-critic best episode return: `{summary.get('gpu_actor_critic_best_episode_return', float('nan')):.3f}`",
        f"- Best robust scaffold model: `{summary.get('best_robust_model_family', 'n/a')}`",
        f"- Best robust scaffold RMSE: `{summary.get('best_robust_scaffold_rmse', float('nan')):.3f}` +/- `{summary.get('best_robust_scaffold_rmse_std', float('nan')):.3f}`",
        f"- Reward-hacking challenge trusted pass rate: `{summary.get('challenge_trusted_pass_rate', float('nan')):.3f}`",
        f"- Reward-hacking challenge proxy demoted rate: `{summary.get('challenge_proxy_demoted_rate', float('nan')):.3f}`",
        f"- Source holdout mean RMSE: `{summary.get('source_holdout_mean_rmse', float('nan')):.3f}`",
        f"- Best source holdout: `{summary.get('source_holdout_best_source', 'n/a')}` with RMSE `{summary.get('source_holdout_best_rmse', float('nan')):.3f}`",
        f"- Source holdout mean recall @ top 20%: `{summary.get('source_holdout_mean_recall_top20pct', float('nan')):.3f}`",
        f"- Rediscovery protected recall @ top 10: `{summary.get('rediscovery_protected_top10_recall', float('nan')):.3f}`",
        f"- Rediscovery naive recall @ top 10: `{summary.get('rediscovery_naive_top10_recall', float('nan')):.3f}`",
        f"- Rediscovery protected recall @ top 20: `{summary.get('rediscovery_protected_top20_recall', float('nan')):.3f}`",
        f"- Rediscovery naive recall @ top 20: `{summary.get('rediscovery_naive_top20_recall', float('nan')):.3f}`",
        "",
        "## Most Demoted By Anti-Hacking Audit",
        _format_table(
            demoted,
            [
                "rank",
                "naive_rank",
                "audit_demote_positions",
                "predicted_pIC50",
                "QED",
                "reward_hacking_risk",
                "audit_status",
            ],
        ),
        "",
        "## Most Promoted By Protected Ranking",
        _format_table(
            promoted,
            [
                "rank",
                "naive_rank",
                "audit_promote_positions",
                "predicted_pIC50",
                "QED",
                "reward_hacking_risk",
                "audit_status",
            ],
        ),
    ]

    if market_df is not None and not market_df.empty:
        lines.extend(
            [
                "",
                "## Marketed Benchmark Snapshot",
                _format_table(
                    market_df,
                    ["name", "predicted_pIC50", "QED", "reward_hacking_risk", "final_score"],
                ),
            ]
        )

    if generated_df is not None and not generated_df.empty:
        lines.extend(
            [
                "",
                "## Generated Candidate Snapshot",
                _format_table(
                    generated_df,
                    ["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "final_score"],
                ),
            ]
        )

    if shortlist_df is not None and not shortlist_df.empty:
        lines.extend(
            [
                "",
                "## Novel Shortlist Snapshot",
                _format_table(
                    shortlist_df,
                    ["smiles", "predicted_pIC50", "QED", "max_market_similarity", "final_score"],
                ),
            ]
        )

    if structural_df is not None and not structural_df.empty:
        lines.extend(
            [
                "",
                "## Structural Rescoring Snapshot",
                _format_table(
                    structural_df,
                    ["smiles", "docking_backend", "vina_affinity_kcal", "interaction_support_score", "docking_rescore", "closest_pose_reference", "final_score"],
                ),
            ]
        )

    if feasibility_df is not None and not feasibility_df.empty:
        lines.extend(
            [
                "",
                "## Feasibility Evidence Snapshot",
                _format_table(
                    feasibility_df,
                    ["smiles", "feasibility_score", "feasibility_status", "max_active_similarity", "fragment_support_ratio"],
                ),
            ]
        )

    if readiness_df is not None and not readiness_df.empty:
        lines.extend(
            [
                "",
                "## Experimental Readiness Snapshot",
                _format_table(
                    readiness_df,
                    [
                        "smiles",
                        "predicted_pIC50",
                        "experimental_readiness_score",
                        "experimental_readiness_status",
                        "experimental_track",
                        "cross_database_consensus_score",
                    ],
                ),
            ]
        )

    if crossdb_df is not None and not crossdb_df.empty:
        lines.extend(
            [
                "",
                "## Cross-Database Validation Snapshot",
                _format_table(
                    crossdb_df,
                    [
                        "smiles",
                        "predicted_pIC50",
                        "cross_database_consensus_score",
                        "external_evidence_support",
                        "external_evidence_status",
                        "cross_database_independent_support_count",
                        "cross_database_status",
                    ],
                ),
            ]
        )

    if prospective_df is not None and not prospective_df.empty:
        lines.extend(
            [
                "",
                "## Prospective Validation Batch",
                _format_table(
                    prospective_df,
                    [
                        "prospective_batch_rank",
                        "candidate_source",
                        "predicted_pIC50",
                        "experimental_readiness_score",
                        "prospective_acquisition_score",
                        "experimental_readiness_status",
                    ],
                ),
            ]
        )

    if rl_df is not None and not rl_df.empty:
        lines.extend(
            [
                "",
                "## Verifiable RL Snapshot",
                _format_table(
                    rl_df,
                    ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "experimental_readiness_score", "rl_priority_score"],
                ),
            ]
        )

    source_holdout_df = _optional_csv(PROJECT_ROOT / "reports" / "source_holdout_benchmark.csv")
    if source_holdout_df is not None and not source_holdout_df.empty:
        lines.extend(
            [
                "",
                "## Source Holdout Benchmark",
                _format_table(
                    source_holdout_df,
                    ["source", "test_size", "rmse", "r2", "rmse_gain_vs_baseline", "recall_top20pct"],
                ),
            ]
        )

    rediscovery_panel_df = _optional_csv(PROJECT_ROOT / "reports" / "rediscovery_benchmark" / "rediscovery_panel.csv")
    if rediscovery_panel_df is not None and not rediscovery_panel_df.empty:
        lines.extend(
            [
                "",
                "## Rediscovery Benchmark",
                _format_table(
                    rediscovery_panel_df[rediscovery_panel_df["benchmark_positive"] == True],
                    ["benchmark_name", "benchmark_source", "protected_panel_rank", "naive_panel_rank", "external_evidence_support", "evidence_arbiter_support"],
                ),
            ]
        )

    gpu_rl_df = _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_dqn" / "gpu_rl_top_candidates.csv")
    if gpu_rl_df is not None and not gpu_rl_df.empty:
        lines.extend(
            [
                "",
                "## GPU DQN RL Snapshot",
                _format_table(
                    gpu_rl_df,
                    ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "evidence_arbiter_support", "gpu_rl_priority_score"],
                ),
            ]
        )

    actor_critic_df = _optional_csv(PROJECT_ROOT / "reports" / "rl_gpu_actor_critic" / "gpu_actor_critic_top_candidates.csv")
    if actor_critic_df is not None and not actor_critic_df.empty:
        lines.extend(
            [
                "",
                "## GPU Actor-Critic Snapshot",
                _format_table(
                    actor_critic_df,
                    ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "experimental_readiness_score", "actor_critic_priority_score"],
                ),
            ]
        )

    (out_dir / "technical_notebook_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_assets(ranked_path: Path | None = None, out_dir: Path | None = None) -> None:
    output_dir = out_dir or NOTEBOOK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = load_csv_artifact(
        ranked_path or (PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"),
        required_columns=REQUIRED_RANKED_COLUMNS,
        producer="python -m src.models.rank_dataset",
    )
    model_summary = _optional_json(PROJECT_ROOT / "reports" / "model_performance_summary.json")
    market_df = _optional_csv(PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv")
    if market_df is None:
        market_df = _optional_csv(PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv")
    generated_df = _optional_csv(PROJECT_ROOT / "reports" / "final_diverse_candidates.csv")
    shortlist_df = _optional_csv(PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv")
    structural_df = _optional_csv(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv")
    feasibility_df = _first_existing_csv(
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_feasibility.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv",
    )
    readiness_df = _first_existing_csv(
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_feasibility.csv",
    )
    crossdb_df = _first_existing_csv(
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_crossdb.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv",
    )
    prospective_df = _optional_csv(PROJECT_ROOT / "reports" / "prospective_validation_batch.csv")
    rl_df = _optional_csv(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_top_candidates.csv")
    rl_summary = _optional_json(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_training_summary.json")
    gpu_gnn_df = _optional_csv(PROJECT_ROOT / "reports" / "gpu_gnn_benchmark.csv")
    pubchem_assay_catalog = _optional_csv(PROJECT_ROOT / "data" / "processed" / "pubchem_egfr_assay_catalog.csv")
    ablation_df = _optional_csv(PROJECT_ROOT / "reports" / "multi_agent_ablation.csv")

    _plot_pipeline_flowchart(output_dir)
    _plot_single_vs_multi_agent(ablation_df, output_dir)
    _plot_risk_distribution(ranked, output_dir)
    _plot_naive_vs_verified(ranked, output_dir)
    _plot_rank_shift(ranked, output_dir)
    _plot_agent_support_heatmap(ranked, output_dir)
    _plot_novelty_vs_applicability(ranked, output_dir)
    _plot_model_split_performance(model_summary, output_dir)
    _plot_gpu_gnn_benchmark(gpu_gnn_df, output_dir)
    _plot_uncertainty_calibration(model_summary, output_dir)
    _plot_structural_rescoring(structural_df, output_dir)
    _plot_vina_affinity(structural_df, output_dir)
    _plot_interaction_support(structural_df, output_dir)
    _plot_readiness_vs_structure(readiness_df, output_dir)
    _plot_cross_database_vs_potency(crossdb_df, output_dir)
    _plot_pubchem_assay_relevance(pubchem_assay_catalog, output_dir)
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility_feasibility_vs_potency.png", output_dir / "feasibility_vs_potency.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_training_curve.png", output_dir / "rl_training_curve.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_reward_breakdown.png", output_dir / "rl_reward_breakdown.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_external_evidence_vs_priority.png", output_dir / "rl_external_evidence_vs_priority.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rl_gpu_dqn" / "gpu_rl_training_curve.png", output_dir / "gpu_rl_training_curve.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rl_gpu_actor_critic" / "gpu_rl_training_curve.png", output_dir / "gpu_actor_critic_training_curve.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "model_robustness_scaffold.png", output_dir / "model_robustness_scaffold.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "reward_hacking_challenge" / "challenge_rank_shift.png", output_dir / "challenge_rank_shift.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "reward_hacking_challenge" / "challenge_status_rates.png", output_dir / "challenge_status_rates.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "source_holdout_rmse.png", output_dir / "source_holdout_rmse.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "source_holdout_recall.png", output_dir / "source_holdout_recall.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rediscovery_benchmark" / "rediscovery_recall_at_k.png", output_dir / "rediscovery_recall_at_k.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "rediscovery_benchmark" / "rediscovery_rank_shift.png", output_dir / "rediscovery_rank_shift.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "cross_database_consensus_vs_readiness.png", output_dir / "cross_database_consensus_vs_readiness.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "cross_database_status_counts.png", output_dir / "cross_database_status_counts.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "external_evidence_support_vs_potency.png", output_dir / "external_evidence_support_vs_potency.png")
    _copy_plot_if_exists(PROJECT_ROOT / "reports" / "prospective_batch_readiness_vs_novelty.png", output_dir / "prospective_batch_readiness_vs_novelty.png")
    _plot_market_comparison(market_df, generated_df, shortlist_df, output_dir)
    _plot_structural_benchmark_boxplots(market_df, structural_df, generated_df, prospective_df, output_dir)
    _plot_generator_benchmark_overview(output_dir)
    _plot_chemical_space_snapshot(ranked, market_df, shortlist_df, output_dir)
    _write_summary(
        ranked,
        market_df,
        generated_df,
        shortlist_df,
        model_summary,
        structural_df,
        feasibility_df,
        readiness_df,
        crossdb_df,
        prospective_df,
        rl_df,
        rl_summary,
        output_dir,
    )

    print(f"[OK] Saved technical notebook assets: {output_dir}")


def main() -> None:
    build_assets()


if __name__ == "__main__":
    main()
