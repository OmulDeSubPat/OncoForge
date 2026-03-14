from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
                    ["smiles", "closest_pose_reference", "docking_rescore", "shape_similarity", "final_score"],
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
    market_df = _optional_csv(PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv")
    generated_df = _optional_csv(PROJECT_ROOT / "reports" / "final_diverse_candidates.csv")
    shortlist_df = _optional_csv(PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv")
    structural_df = _optional_csv(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv")

    _plot_risk_distribution(ranked, output_dir)
    _plot_naive_vs_verified(ranked, output_dir)
    _plot_rank_shift(ranked, output_dir)
    _plot_agent_support_heatmap(ranked, output_dir)
    _plot_novelty_vs_applicability(ranked, output_dir)
    _plot_model_split_performance(model_summary, output_dir)
    _plot_uncertainty_calibration(model_summary, output_dir)
    _plot_structural_rescoring(structural_df, output_dir)
    _plot_market_comparison(market_df, generated_df, shortlist_df, output_dir)
    _plot_chemical_space_snapshot(ranked, market_df, shortlist_df, output_dir)
    _write_summary(ranked, market_df, generated_df, shortlist_df, model_summary, structural_df, output_dir)

    print(f"[OK] Saved technical notebook assets: {output_dir}")


def main() -> None:
    build_assets()


if __name__ == "__main__":
    main()
