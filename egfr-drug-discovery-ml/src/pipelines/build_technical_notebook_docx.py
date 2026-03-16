from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

from src.config import PROJECT_ROOT
from src.visualization.technical_notebook_molecule_views import (
    build_candidate_3d_views,
    build_candidate_grid,
)


REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_DIR = REPORTS_DIR / "technical_notebook"
HISTORY_DIR = REPORTS_DIR / "technical_notebook_history"
DOCX_PATH = REPORTS_DIR / "OncoForge_Technical_Notebook.docx"
HISTORY_INDEX = HISTORY_DIR / "run_history.json"
CONTEXT_MEMORY = HISTORY_DIR / "context_memory.md"

PLOT_ORDER = [
    "risk_distribution_by_audit_status.png",
    "naive_vs_protected_scores.png",
    "audit_rank_demotions.png",
    "top_leads_agent_support_heatmap.png",
    "novelty_vs_applicability.png",
    "model_split_performance.png",
    "gpu_gnn_scaffold_benchmark.png",
    "uncertainty_calibration.png",
    "structural_rescoring_scatter.png",
    "vina_affinity_vs_priority.png",
    "interaction_support_vs_vina.png",
    "readiness_vs_structure.png",
    "cross_database_vs_potency.png",
    "cross_database_consensus_vs_readiness.png",
    "cross_database_status_counts.png",
    "external_evidence_support_vs_potency.png",
    "pubchem_assay_relevance.png",
    "feasibility_vs_potency.png",
    "generator_benchmark_overview.png",
    "rl_training_curve.png",
    "rl_reward_breakdown.png",
    "rl_external_evidence_vs_priority.png",
    "gpu_rl_training_curve.png",
    "model_robustness_scaffold.png",
    "source_holdout_rmse.png",
    "source_holdout_recall.png",
    "rediscovery_recall_at_k.png",
    "rediscovery_rank_shift.png",
    "challenge_rank_shift.png",
    "challenge_status_rates.png",
    "prospective_batch_readiness_vs_novelty.png",
    "marketed_vs_generated_boxplots.png",
    "structural_benchmark_boxplots.png",
    "technical_notebook_chemical_space.png",
]


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _load_df(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select_candidate_source() -> tuple[pd.DataFrame, str]:
    candidates = [
        (REPORTS_DIR / "prospective_validation_batch.csv", "prospective_batch"),
        (REPORTS_DIR / "iterative_ai_optimized_candidates_feasibility.csv", "optimized_feasibility"),
        (REPORTS_DIR / "final_diverse_candidates.csv", "final_diverse"),
        (REPORTS_DIR / "rl_verifiable" / "rl_top_candidates.csv", "verifiable_rl"),
        (REPORTS_DIR / "iterative_ai_optimized_candidates_structural_rescored.csv", "iterative_structural"),
        (REPORTS_DIR / "iterative_ai_optimized_candidates.csv", "iterative"),
    ]
    for path, label in candidates:
        df = _load_df(path)
        if df is not None and not df.empty:
            return df, label
    raise FileNotFoundError("No candidate artifact available for the technical notebook Word export.")


def _current_run_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _candidate_summary(row: pd.Series) -> dict:
    summary = {
        "smiles": str(row.get("smiles", "")),
        "predicted_pIC50": float(row.get("predicted_pIC50", 0.0)),
        "QED": float(row.get("QED", 0.0)),
        "final_score": float(row.get("final_score", 0.0)),
    }
    if pd.notna(row.get("docking_rescore", None)):
        summary["docking_rescore"] = float(row["docking_rescore"])
    if pd.notna(row.get("vina_affinity_kcal", None)):
        summary["vina_affinity_kcal"] = float(row["vina_affinity_kcal"])
    if pd.notna(row.get("feasibility_score", None)):
        summary["feasibility_score"] = float(row["feasibility_score"])
    return summary


def _format_delta(current: float | None, previous: float | None, lower_is_better: bool = False) -> str:
    if current is None or previous is None:
        return "n/a"
    delta = float(current) - float(previous)
    if abs(delta) < 1e-9:
        return "unchanged"
    improved = delta < 0 if lower_is_better else delta > 0
    direction = "improved" if improved else "regressed"
    return f"{direction} by {abs(delta):.3f}"


def _build_evolution_note(snapshot: dict, previous_snapshot: dict | None) -> str:
    if previous_snapshot is None:
        return (
            "This run establishes a new recorded checkpoint for OncoForge. "
            "The notebook starts tracking model quality, candidate strength, structure support, and feasibility from this baseline."
        )

    current_model = snapshot.get("model_summary", {})
    previous_model = previous_snapshot.get("model_summary", {})
    current_metrics = snapshot.get("notebook_metrics", {})
    previous_metrics = previous_snapshot.get("notebook_metrics", {})
    current_top = (snapshot.get("top_candidates") or [{}])[0]
    previous_top = (previous_snapshot.get("top_candidates") or [{}])[0]

    scaffold_note = _format_delta(current_model.get("scaffold_rmse"), previous_model.get("scaffold_rmse"), lower_is_better=True)
    feasibility_note = _format_delta(current_metrics.get("mean_feasibility_score"), previous_metrics.get("mean_feasibility_score"))
    docking_note = _format_delta(current_metrics.get("best_vina_affinity_kcal"), previous_metrics.get("best_vina_affinity_kcal"), lower_is_better=True)
    top_score_note = _format_delta(current_top.get("final_score"), previous_top.get("final_score"))

    return (
        f"Compared with run {previous_snapshot.get('run_label', 'n/a')}, the scaffold validation profile {scaffold_note}, "
        f"mean feasibility {feasibility_note}, strongest structural support {docking_note}, and the top candidate priority score {top_score_note}. "
        f"This short note is kept on purpose so the document shows the evolution of the project without losing earlier iterations."
    )


def _prepare_run_snapshot(run_label: str) -> dict:
    candidate_df, candidate_source = _select_candidate_source()
    model_summary = _load_json(REPORTS_DIR / "model_performance_summary.json") or {}
    notebook_metrics = _load_json(NOTEBOOK_DIR / "technical_notebook_metrics.json") or {}

    run_dir = HISTORY_DIR / run_label
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    top_candidates = candidate_df.head(12).copy().reset_index(drop=True)
    if "rank" not in top_candidates.columns:
        top_candidates["rank"] = top_candidates.index + 1

    grid_path = build_candidate_grid(top_candidates, image_dir / "top_candidates_grid.png")
    top3_assets = []
    for idx, (_, row) in enumerate(top_candidates.head(3).iterrows(), start=1):
        render_path = build_candidate_3d_views(row, image_dir / f"top3_candidate_{idx}_3d.png")
        top3_assets.append(
            {
                "rank": int(row.get("rank", idx)),
                "image": str(render_path) if render_path else None,
                "summary": _candidate_summary(row),
            }
        )

    copied_plots = []
    for plot_name in PLOT_ORDER:
        source = NOTEBOOK_DIR / plot_name
        if not source.exists():
            continue
        target = image_dir / plot_name
        target.write_bytes(source.read_bytes())
        copied_plots.append(str(target))

    rl_candidate_df = _load_df(REPORTS_DIR / "rl_verifiable" / "rl_top_candidates.csv")
    top_candidate_records = [_candidate_summary(row) for _, row in top_candidates.head(5).iterrows()]
    context_notes = [
        f"Candidate source for this run: {candidate_source}.",
        f"Top candidate score={_fmt(top_candidate_records[0].get('final_score') if top_candidate_records else None)}, "
        f"pIC50={_fmt(top_candidate_records[0].get('predicted_pIC50') if top_candidate_records else None)}, "
        f"Vina={_fmt(top_candidate_records[0].get('vina_affinity_kcal') if top_candidate_records else None)}." if top_candidate_records else "No top candidate was available.",
        f"Notebook mean feasibility={_fmt(notebook_metrics.get('mean_feasibility_score'))}, "
        f"best Vina={_fmt(notebook_metrics.get('best_vina_affinity_kcal'))}, "
        f"mean readiness={_fmt(notebook_metrics.get('mean_experimental_readiness'))}, "
        f"broad generator count={_fmt(notebook_metrics.get('generated_candidate_count'))}, "
        f"iterative generator priority={_fmt(notebook_metrics.get('iterative_mean_generator_priority'))}, "
        f"evidence arbiter={_fmt(notebook_metrics.get('evidence_arbiter_mean_support'))}, "
        f"cross-db mean consensus={_fmt(notebook_metrics.get('cross_database_mean_consensus'))}, "
        f"cross-db strong rate={_fmt(notebook_metrics.get('cross_database_strong_rate'))}, "
        f"external evidence support={_fmt(notebook_metrics.get('external_evidence_mean_support'))}, "
        f"RL external evidence={_fmt(notebook_metrics.get('rl_mean_external_evidence_support'))}, "
        f"GPU GNN scaffold RMSE={_fmt(notebook_metrics.get('gpu_gnn_best_scaffold_rmse'))}, "
        f"GPU RL external evidence={_fmt(notebook_metrics.get('gpu_rl_mean_external_evidence_support'))}, "
        f"source holdout RMSE={_fmt(notebook_metrics.get('source_holdout_mean_rmse'))}, "
        f"rediscovery protected top10={_fmt(notebook_metrics.get('rediscovery_protected_top10_recall'))}, "
        f"prospective batch size={_fmt(notebook_metrics.get('prospective_batch_size'))}, "
        f"RL best episode return={_fmt(notebook_metrics.get('rl_best_episode_return'))}.",
    ]
    history = _load_history()
    previous_snapshot = history[-1] if history else None
    evolution_note = _build_evolution_note(snapshot={
        "run_label": run_label,
        "model_summary": {
            "dataset_name": model_summary.get("dataset_name"),
            "random_rmse": model_summary.get("random_split", {}).get("rmse"),
            "scaffold_rmse": model_summary.get("scaffold_split", {}).get("rmse"),
            "temporal_rmse": model_summary.get("temporal_split", {}).get("rmse"),
        },
        "notebook_metrics": notebook_metrics,
        "top_candidates": top_candidate_records,
    }, previous_snapshot=previous_snapshot)
    context_notes.append(evolution_note)
    snapshot = {
        "run_label": run_label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_source": candidate_source,
        "rl_top_candidates": (rl_candidate_df.head(5).to_dict(orient="records") if rl_candidate_df is not None else []),
        "model_summary": {
            "dataset_name": model_summary.get("dataset_name"),
            "random_rmse": model_summary.get("random_split", {}).get("rmse"),
            "scaffold_rmse": model_summary.get("scaffold_split", {}).get("rmse"),
            "temporal_rmse": model_summary.get("temporal_split", {}).get("rmse"),
        },
        "notebook_metrics": notebook_metrics,
        "top_candidates": top_candidate_records,
        "grid_image": str(grid_path) if grid_path else None,
        "top3_assets": top3_assets,
        "plots": copied_plots,
        "context_notes": context_notes,
        "evolution_note": evolution_note,
    }
    (run_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def _load_history() -> list[dict]:
    if not HISTORY_INDEX.exists():
        return []
    return json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))


def _save_history(history: list[dict]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_INDEX.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_context_memory(snapshot: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## {snapshot.get('run_label', 'unknown_run')}",
        f"- Created at: {snapshot.get('created_at', 'n/a')}",
        f"- Candidate source: {snapshot.get('candidate_source', 'n/a')}",
    ]
    for note in snapshot.get("context_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    with CONTEXT_MEMORY.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _upsert_snapshot(history: list[dict], snapshot: dict) -> list[dict]:
    filtered = [entry for entry in history if entry.get("run_label") != snapshot.get("run_label")]
    filtered.append(snapshot)
    filtered.sort(key=lambda entry: entry.get("created_at", ""))
    return filtered


def _add_title(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = document.styles["Title"].font.size


def _add_bullet(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Bullet")


def _add_picture_if_exists(document: Document, path_str: str | None, width: float = 6.3) -> None:
    if not path_str:
        return
    path = Path(path_str)
    if not path.exists():
        return
    document.add_picture(str(path), width=Inches(width))


def _add_run_comparison(document: Document, history: list[dict]) -> None:
    if not history:
        return
    document.add_heading("Run-to-Run Comparison", level=1)
    table = document.add_table(rows=1, cols=6)
    header = table.rows[0].cells
    header[0].text = "Run"
    header[1].text = "Scaffold RMSE"
    header[2].text = "Mean Feasibility"
    header[3].text = "Best Vina"
    header[4].text = "Top Score"
    header[5].text = "Top pIC50"

    for snapshot in history:
        row = table.add_row().cells
        metrics = snapshot.get("notebook_metrics", {})
        top_candidate = (snapshot.get("top_candidates") or [{}])[0]
        row[0].text = str(snapshot.get("run_label", "n/a"))
        row[1].text = _fmt(snapshot.get("model_summary", {}).get("scaffold_rmse"))
        row[2].text = _fmt(metrics.get("mean_feasibility_score"))
        row[3].text = _fmt(metrics.get("best_vina_affinity_kcal"))
        row[4].text = _fmt(top_candidate.get("final_score"))
        row[5].text = _fmt(top_candidate.get("predicted_pIC50"))


def _build_docx(history: list[dict]) -> Path:
    document = Document()
    _add_title(
        document,
        "OncoForge: A Multi-Agent Reinforcement Learning AI with Verifiable Rewards for Generating and Evaluating Anticancer Molecules",
    )
    document.add_paragraph(
        "Technical notebook generated automatically from major pipeline runs. "
        "This document tracks model quality, reward-audit behavior, visualization outputs, and the strongest candidate molecules over time."
    )
    _add_run_comparison(document, history)

    for snapshot in history:
        document.add_section(WD_SECTION.NEW_PAGE)
        document.add_heading(f"Run {snapshot['run_label']}", level=1)
        document.add_paragraph(f"Created at: {snapshot.get('created_at', 'n/a')}")
        document.add_paragraph(f"Candidate source: {snapshot.get('candidate_source', 'n/a')}")

        model_summary = snapshot.get("model_summary", {})
        document.add_heading("Run Summary", level=2)
        _add_bullet(document, f"Random RMSE: {model_summary.get('random_rmse', 'n/a')}")
        _add_bullet(document, f"Scaffold RMSE: {model_summary.get('scaffold_rmse', 'n/a')}")
        _add_bullet(document, f"Temporal RMSE: {model_summary.get('temporal_rmse', 'n/a')}")
        notebook_metrics = snapshot.get("notebook_metrics", {})
        _add_bullet(document, f"Audit pass rate: {notebook_metrics.get('audit_pass_rate', 'n/a')}")
        _add_bullet(document, f"Audit fail rate: {notebook_metrics.get('audit_fail_rate', 'n/a')}")
        _add_bullet(document, f"Median reward hacking risk: {notebook_metrics.get('median_reward_hacking_risk', 'n/a')}")
        _add_bullet(document, f"Mean feasibility score: {notebook_metrics.get('mean_feasibility_score', 'n/a')}")
        _add_bullet(document, f"Mean experimental readiness: {notebook_metrics.get('mean_experimental_readiness', 'n/a')}")
        _add_bullet(document, f"Evidence arbiter mean support: {notebook_metrics.get('evidence_arbiter_mean_support', 'n/a')}")
        _add_bullet(document, f"Evidence arbiter pass rate: {notebook_metrics.get('evidence_arbiter_pass_rate', 'n/a')}")
        _add_bullet(document, f"Cross-database mean consensus: {notebook_metrics.get('cross_database_mean_consensus', 'n/a')}")
        _add_bullet(document, f"Cross-database strong rate: {notebook_metrics.get('cross_database_strong_rate', 'n/a')}")
        _add_bullet(document, f"External evidence mean support: {notebook_metrics.get('external_evidence_mean_support', 'n/a')}")
        _add_bullet(document, f"Papyrus molecules: {notebook_metrics.get('papyrus_unique_molecules', 'n/a')}")
        _add_bullet(document, f"ExCAPE molecules: {notebook_metrics.get('excape_unique_molecules', 'n/a')}")
        _add_bullet(document, f"PubChem mean enriched evidence: {notebook_metrics.get('pubchem_mean_enriched_evidence_score', 'n/a')}")
        _add_bullet(document, f"PubChem strong evidence rate: {notebook_metrics.get('pubchem_strong_evidence_rate', 'n/a')}")
        _add_bullet(document, f"RL mean external evidence support: {notebook_metrics.get('rl_mean_external_evidence_support', 'n/a')}")
        _add_bullet(document, f"RL ready rate: {notebook_metrics.get('rl_readiness_ready_rate', 'n/a')}")
        _add_bullet(document, f"GPU GNN best scaffold model: {notebook_metrics.get('gpu_gnn_best_scaffold_model', 'n/a')}")
        _add_bullet(document, f"GPU GNN best scaffold RMSE: {notebook_metrics.get('gpu_gnn_best_scaffold_rmse', 'n/a')}")
        _add_bullet(document, f"GPU RL mean external evidence support: {notebook_metrics.get('gpu_rl_mean_external_evidence_support', 'n/a')}")
        _add_bullet(document, f"GPU RL best episode return: {notebook_metrics.get('gpu_rl_best_episode_return', 'n/a')}")
        _add_bullet(document, f"Source holdout mean RMSE: {notebook_metrics.get('source_holdout_mean_rmse', 'n/a')}")
        _add_bullet(document, f"Source holdout best source: {notebook_metrics.get('source_holdout_best_source', 'n/a')}")
        _add_bullet(document, f"Rediscovery protected top-10 recall: {notebook_metrics.get('rediscovery_protected_top10_recall', 'n/a')}")
        _add_bullet(document, f"Rediscovery protected top-20 recall: {notebook_metrics.get('rediscovery_protected_top20_recall', 'n/a')}")
        _add_bullet(document, f"Best Vina affinity: {notebook_metrics.get('best_vina_affinity_kcal', 'n/a')}")
        _add_bullet(document, f"Prospective batch size: {notebook_metrics.get('prospective_batch_size', 'n/a')}")

        document.add_heading("Context Notes", level=2)
        if snapshot.get("evolution_note"):
            document.add_paragraph(snapshot["evolution_note"])
        for note in snapshot.get("context_notes", []):
            if note == snapshot.get("evolution_note"):
                continue
            _add_bullet(document, note)

        document.add_heading("Top Molecules", level=2)
        _add_picture_if_exists(document, snapshot.get("grid_image"))
        for idx, candidate in enumerate(snapshot.get("top_candidates", []), start=1):
            document.add_paragraph(
                f"Top {idx}: score={_fmt(candidate.get('final_score'))}, "
                f"pIC50={_fmt(candidate.get('predicted_pIC50'))}, "
                f"QED={_fmt(candidate.get('QED'))}"
            )
            if "feasibility_score" in candidate:
                document.add_paragraph(f"Feasibility score: {_fmt(candidate.get('feasibility_score'))}")
            if "vina_affinity_kcal" in candidate:
                document.add_paragraph(f"Vina affinity: {_fmt(candidate.get('vina_affinity_kcal'))} kcal/mol")
            document.add_paragraph(candidate.get("smiles", ""))

        if snapshot.get("rl_top_candidates"):
            document.add_heading("Verifiable RL Snapshot", level=2)
            for idx, candidate in enumerate(snapshot.get("rl_top_candidates", [])[:3], start=1):
                document.add_paragraph(
                    f"RL {idx}: score={_fmt(candidate.get('rl_priority_score'))}, "
                    f"pIC50={_fmt(candidate.get('predicted_pIC50'))}, "
                    f"cross-db={_fmt(candidate.get('cross_database_consensus_score'))}, "
                    f"external evidence={_fmt(candidate.get('external_evidence_support'))}, "
                    f"readiness={_fmt(candidate.get('experimental_readiness_score'))}"
                )
                document.add_paragraph(candidate.get("smiles", ""))

        document.add_heading("3D Candidate Views", level=2)
        for asset in snapshot.get("top3_assets", []):
            summary = asset.get("summary", {})
            document.add_paragraph(
                f"Candidate rank {asset.get('rank', '?')} | "
                f"score={_fmt(summary.get('final_score'))} | "
                f"pIC50={_fmt(summary.get('predicted_pIC50'))} | "
                f"QED={_fmt(summary.get('QED'))}"
            )
            if "docking_rescore" in summary:
                document.add_paragraph(f"Structural rescoring support: {_fmt(summary.get('docking_rescore'))}")
            if "vina_affinity_kcal" in summary:
                document.add_paragraph(f"Vina affinity: {_fmt(summary.get('vina_affinity_kcal'))} kcal/mol")
            if "feasibility_score" in summary:
                document.add_paragraph(f"Feasibility score: {_fmt(summary.get('feasibility_score'))}")
            document.add_paragraph(summary.get("smiles", ""))
            _add_picture_if_exists(document, asset.get("image"), width=6.6)

        document.add_heading("Run Visualizations", level=2)
        for plot_path in snapshot.get("plots", []):
            plot_name = Path(plot_path).name.replace("_", " ").replace(".png", "")
            document.add_paragraph(plot_name.title())
            _add_picture_if_exists(document, plot_path)

    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(DOCX_PATH))
    return DOCX_PATH


def main(argv: list[str] | None = None) -> None:
    run_label = _current_run_label()
    snapshot = _prepare_run_snapshot(run_label)
    history = _load_history()
    history = _upsert_snapshot(history, snapshot)
    _save_history(history)
    _append_context_memory(snapshot)
    out_path = _build_docx(history)
    print(f"[OK] Saved technical notebook Word document: {out_path}")
    print(f"[OK] Updated run history: {HISTORY_INDEX}")


if __name__ == "__main__":
    main()
