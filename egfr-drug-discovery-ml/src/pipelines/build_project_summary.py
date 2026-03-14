from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_top_table(df: pd.DataFrame | None, columns: list[str], n: int = 5) -> str:
    if df is None or df.empty:
        return "_Artifact missing or empty._"
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return "_Expected columns are not available in this artifact yet._"
    subset = df[available_columns].head(n).copy()

    header = "| " + " | ".join(available_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(available_columns)) + " |"
    rows = [header, separator]

    for _, row in subset.iterrows():
        formatted = [str(row[column]) for column in available_columns]
        rows.append("| " + " | ".join(formatted) + " |")

    return "\n".join(rows)


def main():
    reports_dir = PROJECT_ROOT / "reports"
    summary_path = reports_dir / "isef_project_summary.md"
    notebook_metrics_path = reports_dir / "technical_notebook" / "technical_notebook_metrics.json"
    notebook_metrics_quick_path = reports_dir / "technical_notebook_quick" / "technical_notebook_metrics.json"

    metrics = _load_json(reports_dir / "model_performance_summary.json") or {}
    notebook_metrics = _load_json(notebook_metrics_path) or _load_json(notebook_metrics_quick_path) or {}
    ranked_df = _load_csv(reports_dir / "ranked_egfr_dataset.csv")
    market_df = _load_csv(reports_dir / "marketed_egfr_scored.csv")
    diverse_df = _load_csv(reports_dir / "final_diverse_candidates.csv")
    shortlist_df = _load_csv(reports_dir / "market_comparable_novel_shortlist.csv")

    random_metrics = metrics.get("random_split", {})
    scaffold_metrics = metrics.get("scaffold_split", {})

    lines = [
        "# OncoForge ISEF Summary",
        "",
        "## Project Goal",
        "OncoForge is an AI-assisted lead-optimization pipeline for EGFR inhibitors.",
        "The system does not claim to discover finished drugs; it prioritizes chemically plausible, high-potential candidates for downstream wet-lab validation.",
        "",
        "## Upgraded Methodology",
        "- A multi-agent scorer now separates potency, chemistry, safety, novelty and applicability-domain checks instead of relying on a single scalar reward.",
        "- Verified reward is combined with anti-reward-hacking audits so suspicious molecules are penalized even if they exploit a proxy metric.",
        "- Ranking now uses multi-objective percentiles, veto logic and diversity-aware post-filtering.",
        "- Ensemble training is evaluated on both random and scaffold splits, then retrained on the full dataset for final inference.",
        "",
        "## Why The Multi-Agent Design Helps",
        "- `Potency agent`: predicts pIC50 with ensemble uncertainty.",
        "- `Chemistry agent`: scores QED, SA, Lipinski pressure and descriptor sanity.",
        "- `Safety agent`: checks PAINS and structural alerts.",
        "- `Novelty/applicability agent`: balances novelty against the training distribution and marketed drugs.",
        "- `Audit agent`: flags reward-hacking patterns such as highly potent but out-of-domain or unsafe structures.",
        "- `Protected ranker`: compares a naive proxy score against a protected score and explicitly demotes suspicious molecules.",
        "",
        "## Model Performance",
        f"- Dataset size: `{metrics.get('dataset_size', 'n/a')}` molecules",
        f"- Random split RMSE / R2: `{random_metrics.get('rmse', 'n/a')}` / `{random_metrics.get('r2', 'n/a')}`",
        f"- Scaffold split RMSE / R2: `{scaffold_metrics.get('rmse', 'n/a')}` / `{scaffold_metrics.get('r2', 'n/a')}`",
        "",
        "## Audit Diagnostics",
        f"- Audit pass rate: `{notebook_metrics.get('audit_pass_rate', 'n/a')}`",
        f"- Audit review rate: `{notebook_metrics.get('audit_review_rate', 'n/a')}`",
        f"- Audit fail rate: `{notebook_metrics.get('audit_fail_rate', 'n/a')}`",
        f"- Median reward hacking risk: `{notebook_metrics.get('median_reward_hacking_risk', 'n/a')}`",
        f"- Mean audit demotion: `{notebook_metrics.get('mean_audit_demotion', 'n/a')}` positions",
        "",
        "## Top Ranked Training-Space Molecules",
        _format_top_table(
            ranked_df,
            ["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "audit_status", "final_score"],
        ),
        "",
        "## Scored Marketed EGFR Drugs",
        _format_top_table(
            market_df,
            ["name", "predicted_pIC50", "QED", "reward_hacking_risk", "audit_status", "final_score"],
        ),
        "",
        "## Diverse Generated Candidates",
        _format_top_table(
            diverse_df,
            ["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "audit_status", "final_score"],
        ),
        "",
        "## Market-Comparable Novel Shortlist",
        _format_top_table(
            shortlist_df,
            ["smiles", "predicted_pIC50", "QED", "max_market_similarity", "audit_status", "final_score"],
        ),
        "",
        "## Main Artifacts",
        "- `reports/model_performance_summary.json`",
        "- `reports/ranked_egfr_dataset.csv`",
        "- `reports/marketed_egfr_scored.csv`",
        "- `reports/generated_analogs_ranked.csv`",
        "- `reports/iterative_ai_optimized_candidates.csv`",
        "- `reports/final_diverse_candidates.csv`",
        "- `reports/market_comparable_novel_shortlist.csv`",
        "- `reports/technical_notebook/technical_notebook_summary.md`",
        "- `reports/technical_notebook/technical_notebook_metrics.json`",
        "- `reports/technical_notebook_quick/technical_notebook_summary.md`",
        "- `reports/technical_notebook_quick/technical_notebook_metrics.json`",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved project summary: {summary_path}")


if __name__ == "__main__":
    main()
