from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.knowledge import COMPETITION_LITERATURE, PROJECT_PHASES


REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_DIR = REPORTS_DIR / "technical_notebook"
HISTORY_INDEX = REPORTS_DIR / "technical_notebook_history" / "run_history.json"
CONTEXT_MD = NOTEBOOK_DIR / "competition_report_context.md"
CONTEXT_JSON = NOTEBOOK_DIR / "competition_report_context.json"
LITERATURE_MD = NOTEBOOK_DIR / "literature_review.md"
REFERENCE_CSV = NOTEBOOK_DIR / "reference_library.csv"
ITERATION_CSV = NOTEBOOK_DIR / "project_iteration_history.csv"


def _optional_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_run_history() -> list[dict]:
    history = _optional_json(HISTORY_INDEX)
    return history if isinstance(history, list) else []


def _load_git_log() -> list[dict]:
    repo_root = PROJECT_ROOT.parent
    command = [
        "git",
        "-C",
        str(repo_root),
        "log",
        "--pretty=format:%h|%ad|%s",
        "--date=short",
        "--reverse",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except Exception:
        return []

    rows = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split("|", 2)
        if len(parts) != 3:
            continue
        commit, date_label, subject = parts
        rows.append({"commit": commit, "date": date_label, "subject": subject})
    return rows


def _current_context() -> dict:
    model_summary = _optional_json(REPORTS_DIR / "model_performance_summary.json") or {}
    notebook_metrics = _optional_json(NOTEBOOK_DIR / "technical_notebook_metrics.json") or {}
    multisource_summary = _optional_json(PROJECT_ROOT / "data" / "processed" / "egfr_multisource_summary.json") or {}
    run_history = _load_run_history()
    git_log = _load_git_log()

    dataset_scale_rows = [
        {
            "label": "OncoForge model set",
            "value": int(model_summary.get("dataset_size", 0) or 0),
            "note": "Current cleaned multisource training set",
        },
        {
            "label": "OncoForge evidence pool",
            "value": int(multisource_summary.get("unique_molecules_before_final_cleaning", 0) or 0),
            "note": "Unique molecules before final reduction",
        },
        {
            "label": "Nada 2023 EGFR set",
            "value": 9000,
            "note": "Approximate curated EGFR compounds",
        },
        {
            "label": "DeepEGFR 2025 set",
            "value": 8263,
            "note": "Final curated EGFR classifier set",
        },
    ]

    regression_rows = [
        {
            "label": "OncoForge random",
            "value": _safe_float(model_summary.get("random_split", {}).get("r2")),
            "note": "Current random split R2",
        },
        {
            "label": "OncoForge scaffold",
            "value": _safe_float(model_summary.get("scaffold_split", {}).get("r2")),
            "note": "Current scaffold split R2",
        },
    ]
    for entry in COMPETITION_LITERATURE:
        if entry.comparison_axis == "regression_r2" and entry.comparison_value is not None:
            regression_rows.append(
                {
                    "label": entry.comparison_label or entry.key,
                    "value": float(entry.comparison_value),
                    "note": entry.comparison_note or "",
                }
            )

    classification_rows = []
    for entry in COMPETITION_LITERATURE:
        if entry.comparison_axis == "classification_f1" and entry.comparison_value is not None:
            classification_rows.append(
                {
                    "label": entry.comparison_label or entry.key,
                    "value": float(entry.comparison_value),
                    "note": entry.comparison_note or "",
                }
            )

    context = {
        "model_dataset_size": int(model_summary.get("dataset_size", 0) or 0),
        "ranked_molecules": int(notebook_metrics.get("ranked_molecules", 0) or 0),
        "scaffold_rmse": _safe_float(model_summary.get("scaffold_split", {}).get("rmse")),
        "scaffold_r2": _safe_float(model_summary.get("scaffold_split", {}).get("r2")),
        "random_rmse": _safe_float(model_summary.get("random_split", {}).get("rmse")),
        "random_r2": _safe_float(model_summary.get("random_split", {}).get("r2")),
        "best_vina_affinity_kcal": _safe_float(notebook_metrics.get("best_vina_affinity_kcal")),
        "mean_feasibility_score": _safe_float(notebook_metrics.get("mean_feasibility_score")),
        "cross_database_mean_consensus": _safe_float(notebook_metrics.get("cross_database_mean_consensus")),
        "prospective_batch_size": int(notebook_metrics.get("prospective_batch_size", 0) or 0),
        "git_log": git_log,
        "run_history_count": len(run_history),
        "project_phases": [phase.__dict__ for phase in PROJECT_PHASES],
        "dataset_scale_rows": dataset_scale_rows,
        "regression_rows": regression_rows,
        "classification_rows": classification_rows,
        "literature_count": len(COMPETITION_LITERATURE),
    }
    return context


def _plot_project_evolution(history: list[dict], out_dir: Path) -> None:
    if not history:
        return

    rows = []
    for index, snapshot in enumerate(history, start=1):
        metrics = snapshot.get("notebook_metrics", {})
        top_candidate = (snapshot.get("top_candidates") or [{}])[0]
        rows.append(
            {
                "run_index": index,
                "run_label": snapshot.get("run_label", f"run_{index}"),
                "created_at": snapshot.get("created_at", ""),
                "candidate_source": snapshot.get("candidate_source", "n/a"),
                "scaffold_rmse": _safe_float(snapshot.get("model_summary", {}).get("scaffold_rmse")),
                "mean_feasibility_score": _safe_float(metrics.get("mean_feasibility_score")),
                "best_vina_affinity_kcal": _safe_float(metrics.get("best_vina_affinity_kcal")),
                "top_final_score": _safe_float(top_candidate.get("final_score")),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ITERATION_CSV, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    x = df["run_index"].to_numpy()

    axes[0, 0].plot(x, df["scaffold_rmse"], marker="o", color="#1d3557")
    axes[0, 0].set_title("Scaffold RMSE Across Recorded Runs")
    axes[0, 0].set_xlabel("Recorded notebook run")
    axes[0, 0].set_ylabel("RMSE")

    axes[0, 1].plot(x, df["mean_feasibility_score"], marker="o", color="#2a9d8f")
    axes[0, 1].set_title("Mean Feasibility Score Across Runs")
    axes[0, 1].set_xlabel("Recorded notebook run")
    axes[0, 1].set_ylabel("Feasibility score")

    axes[1, 0].plot(x, df["best_vina_affinity_kcal"], marker="o", color="#e76f51")
    axes[1, 0].set_title("Best Vina Affinity Across Runs")
    axes[1, 0].set_xlabel("Recorded notebook run")
    axes[1, 0].set_ylabel("kcal/mol")

    axes[1, 1].plot(x, df["top_final_score"], marker="o", color="#6d597a")
    axes[1, 1].set_title("Top Candidate Final Score Across Runs")
    axes[1, 1].set_xlabel("Recorded notebook run")
    axes[1, 1].set_ylabel("Final score")

    fig.suptitle("OncoForge Evolution Across Technical Notebook Checkpoints")
    _save_figure(fig, out_dir / "project_evolution_history.png")


def _plot_phase_capability_matrix(out_dir: Path) -> None:
    phases = [phase for phase in PROJECT_PHASES if phase.phase_id != "V0"]
    if not phases:
        return

    capabilities = [
        "Single-source QSAR",
        "Analog generation",
        "Multi-agent audit",
        "Structural rescoring",
        "Cross-database evidence",
        "Prospective batch",
        "Reward-hacking benchmark",
        "Rediscovery benchmark",
        "RL branch",
        "GPU branch",
        "Auto DOCX export",
    ]
    matrix = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(capabilities)))
    ax.set_xticklabels(capabilities, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(phases)))
    ax.set_yticklabels([f"{phase.phase_id}: {phase.title}" for phase in phases])
    ax.set_title("Capability Growth From Baseline Desktop Version to Current OncoForge")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                "Yes" if matrix[row_idx, col_idx] >= 0.5 else "No",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Capability present")
    _save_figure(fig, out_dir / "project_phase_capability_matrix.png")


def _plot_literature_context(context: dict, out_dir: Path) -> None:
    regression_rows = [row for row in context.get("regression_rows", []) if not np.isnan(row["value"])]
    dataset_rows = [row for row in context.get("dataset_scale_rows", []) if row.get("value")]
    classification_rows = [row for row in context.get("classification_rows", []) if not np.isnan(row["value"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    if regression_rows:
        labels = [row["label"] for row in regression_rows]
        values = [row["value"] for row in regression_rows]
        axes[0].barh(labels, values, color=["#1d3557", "#457b9d", "#2a9d8f", "#e9c46a"][: len(labels)])
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_title("EGFR Regression Context (R2)")
        axes[0].set_xlabel("Reported R2")
    else:
        axes[0].axis("off")

    if dataset_rows:
        labels = [row["label"] for row in dataset_rows]
        values = [row["value"] for row in dataset_rows]
        axes[1].barh(labels, values, color=["#6d597a", "#b56576", "#e56b6f", "#eaac8b"][: len(labels)])
        axes[1].set_title("Dataset Scale Context")
        axes[1].set_xlabel("Molecules")
    else:
        axes[1].axis("off")

    if classification_rows:
        labels = [row["label"] for row in classification_rows]
        values = [row["value"] for row in classification_rows]
        axes[2].barh(labels, values, color=["#2a9d8f", "#264653"][: len(labels)])
        axes[2].set_xlim(0.0, 1.0)
        axes[2].set_title("Recent EGFR Classification Context")
        axes[2].set_xlabel("Reported F1")
    else:
        axes[2].axis("off")

    fig.suptitle("OncoForge in Context of Related EGFR AI Studies")
    _save_figure(fig, out_dir / "literature_context_comparison.png")


def _write_reference_csv() -> None:
    rows = []
    for entry in COMPETITION_LITERATURE:
        rows.append(
            {
                "key": entry.key,
                "title": entry.title,
                "citation": entry.citation,
                "url": entry.url,
                "category": entry.category,
                "why_it_matters": entry.why_it_matters,
                "short_quote": entry.short_quote,
                "comparison_axis": entry.comparison_axis or "",
                "comparison_label": entry.comparison_label or "",
                "comparison_value": entry.comparison_value if entry.comparison_value is not None else "",
                "comparison_unit": entry.comparison_unit or "",
                "comparison_note": entry.comparison_note or "",
            }
        )
    pd.DataFrame(rows).to_csv(REFERENCE_CSV, index=False)


def _write_literature_review() -> None:
    grouped: dict[str, list] = defaultdict(list)
    for entry in COMPETITION_LITERATURE:
        grouped[entry.category].append(entry)

    lines = [
        "# Competition Literature Review",
        "",
        "This file is generated for the ISEF-style notebook and groups the main papers used to position OncoForge.",
        "Direct quotes are intentionally short so the document stays readable and compliant.",
    ]
    for category in sorted(grouped):
        lines.extend(["", f"## {category}"])
        for entry in grouped[category]:
            lines.append(f"- {entry.citation} {entry.title}")
            lines.append(f"  Source: {entry.url}")
            lines.append(f"  Why it matters here: {entry.why_it_matters}")
            lines.append(f'  Short quote: "{entry.short_quote}"')
            if entry.comparison_value is not None:
                lines.append(
                    f"  Comparison signal: {entry.comparison_label} = {entry.comparison_value} {entry.comparison_unit or ''}".rstrip()
                )
    LITERATURE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_context_markdown(context: dict) -> None:
    lines = [
        "# Competition Report Context",
        "",
        "## Positioning",
        "OncoForge should be presented as an audited, evidence-aware lead-prioritization platform for EGFR inhibitors.",
        "The strongest claims are about generalization checks, reward protection, rediscovery, and shortlist quality, not about declaring finished drug discovery.",
        "",
        "## Version Narrative",
    ]
    for phase in PROJECT_PHASES:
        lines.append(f"- {phase.phase_id} ({phase.date_label}, {phase.commit}): {phase.title}. {phase.focus}.")
        for upgrade in phase.upgrades:
            lines.append(f"  {upgrade}")

    lines.extend(
        [
            "",
            "## Current Snapshot",
            f"- Model dataset size: `{context.get('model_dataset_size', 0)}` molecules.",
            f"- Ranked molecules: `{context.get('ranked_molecules', 0)}`.",
            f"- Random RMSE / R2: `{context.get('random_rmse', float('nan')):.3f}` / `{context.get('random_r2', float('nan')):.3f}`.",
            f"- Scaffold RMSE / R2: `{context.get('scaffold_rmse', float('nan')):.3f}` / `{context.get('scaffold_r2', float('nan')):.3f}`.",
            f"- Mean feasibility score: `{context.get('mean_feasibility_score', float('nan')):.3f}`.",
            f"- Cross-database mean consensus: `{context.get('cross_database_mean_consensus', float('nan')):.3f}`.",
            f"- Best Vina affinity: `{context.get('best_vina_affinity_kcal', float('nan')):.3f}` kcal/mol.",
            f"- Prospective batch size: `{context.get('prospective_batch_size', 0)}`.",
            f"- Recorded notebook iterations: `{context.get('run_history_count', 0)}`.",
            "",
            "## Comparison Caveat",
            "External study bars are context plots, not strict leaderboard claims.",
            "The datasets, endpoints, split strategies, and evaluation tasks differ across papers, so the safest claim is that OncoForge operates in a performance range that is credible for related EGFR studies while using harder evidence-aware validation than many narrower baselines.",
        ]
    )
    CONTEXT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_assets(out_dir: Path | None = None) -> None:
    output_dir = out_dir or NOTEBOOK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_run_history()
    context = _current_context()

    _plot_project_evolution(history, output_dir)
    _plot_phase_capability_matrix(output_dir)
    _plot_literature_context(context, output_dir)
    _write_reference_csv()
    _write_literature_review()
    _write_context_markdown(context)
    CONTEXT_JSON.write_text(json.dumps(context, indent=2), encoding="utf-8")

    print(f"[OK] Saved competition report assets: {output_dir}")


def main() -> None:
    build_assets()


if __name__ == "__main__":
    main()
