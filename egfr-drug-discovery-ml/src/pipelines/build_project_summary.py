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


def _first_existing_csv(*paths: Path) -> pd.DataFrame | None:
    for path in paths:
        df = _load_csv(path)
        if df is not None:
            return df
    return None


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
    market_df = _load_csv(reports_dir / "marketed_egfr_structural_benchmark.csv")
    if market_df is None:
        market_df = _load_csv(reports_dir / "marketed_egfr_scored.csv")
    diverse_df = _load_csv(reports_dir / "final_diverse_candidates.csv")
    shortlist_df = _load_csv(reports_dir / "market_comparable_novel_shortlist.csv")
    feasibility_df = _first_existing_csv(
        reports_dir / "iterative_ai_optimized_candidates_structural_feasibility.csv",
        reports_dir / "iterative_ai_optimized_candidates_feasibility.csv",
    )
    readiness_df = _first_existing_csv(
        reports_dir / "iterative_ai_optimized_candidates_readiness.csv",
        reports_dir / "iterative_ai_optimized_candidates_structural_feasibility.csv",
    )
    crossdb_df = _first_existing_csv(
        reports_dir / "iterative_ai_optimized_candidates_structural_crossdb.csv",
        reports_dir / "iterative_ai_optimized_candidates_crossdb.csv",
    )
    prospective_df = _load_csv(reports_dir / "prospective_validation_batch.csv")
    generated_summary = _load_json(reports_dir / "generated_analogs_ranked.summary.json") or {}
    ai_guided_summary = _load_json(reports_dir / "ai_guided_analogs.summary.json") or {}
    iterative_summary = _load_json(reports_dir / "iterative_ai_optimized_candidates.summary.json") or {}
    generation_suite_df = _load_csv(reports_dir / "generation_benchmark_suite.csv")
    ablation_df = _load_csv(reports_dir / "studii_ablatie" / "studii_ablatie.csv")
    rl_df = _load_csv(reports_dir / "rl_verifiable" / "rl_top_candidates.csv")
    rl_summary = _load_json(reports_dir / "rl_verifiable" / "rl_training_summary.json") or {}
    gpu_gnn_summary = _load_json(reports_dir / "gpu_gnn_performance_summary.json") or {}
    gpu_rl_df = _load_csv(reports_dir / "rl_gpu_dqn" / "gpu_rl_top_candidates.csv")
    gpu_rl_summary = _load_json(reports_dir / "rl_gpu_dqn" / "gpu_rl_training_summary.json") or {}
    actor_critic_df = _load_csv(reports_dir / "rl_gpu_actor_critic" / "gpu_actor_critic_top_candidates.csv")
    actor_critic_summary = _load_json(reports_dir / "rl_gpu_actor_critic" / "gpu_actor_critic_summary.json") or {}
    pubchem_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "pubchem_egfr_reference.summary.json") or {}
    papyrus_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "papyrus_egfr_reference.summary.json") or {}
    excape_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "excape_egfr_reference.summary.json") or {}
    robustness_df = _load_csv(reports_dir / "model_robustness_summary.csv")
    challenge_df = _load_csv(reports_dir / "reward_hacking_challenge" / "reward_hacking_challenge_summary.csv")
    source_holdout_df = _load_csv(reports_dir / "source_holdout_benchmark.csv")
    rediscovery_summary = _load_json(reports_dir / "rediscovery_benchmark" / "rediscovery_summary.json") or {}

    random_metrics = metrics.get("random_split", {})
    scaffold_metrics = metrics.get("scaffold_split", {})
    temporal_metrics = metrics.get("temporal_split", {})
    feasibility_pass_rate = (
        float((feasibility_df["feasibility_status"] == "pass").mean())
        if feasibility_df is not None and not feasibility_df.empty and "feasibility_status" in feasibility_df.columns
        else "n/a"
    )
    best_vina_affinity = (
        float(feasibility_df["vina_affinity_kcal"].dropna().min())
        if feasibility_df is not None and not feasibility_df.empty and "vina_affinity_kcal" in feasibility_df.columns and not feasibility_df["vina_affinity_kcal"].dropna().empty
        else "n/a"
    )
    robust_line = "n/a"
    if robustness_df is not None and not robustness_df.empty:
        scaffold_rows = robustness_df[robustness_df["split"] == "scaffold"].sort_values("robustness_score")
        if not scaffold_rows.empty:
            best_row = scaffold_rows.iloc[0]
            robust_line = f"{best_row['model_family']} | RMSE {best_row['mean_rmse']:.3f} +/- {best_row['std_rmse']:.3f}"
    challenge_line = "n/a"
    if challenge_df is not None and not challenge_df.empty:
        exploit = challenge_df[challenge_df["cohort"] == "proxy_exploits"]
        trusted = challenge_df[challenge_df["cohort"] == "trusted_controls"]
        if not exploit.empty and not trusted.empty:
            challenge_line = (
                f"trusted pass {float(trusted.iloc[0]['audit_pass_rate']):.3f}, "
                f"proxy demoted>=20 {float(exploit.iloc[0]['demoted_20plus_rate']):.3f}, "
                f"proxy review/fail {float(exploit.iloc[0]['review_or_fail_rate']):.3f}"
            )
    source_holdout_line = "n/a"
    if source_holdout_df is not None and not source_holdout_df.empty:
        best_row = source_holdout_df.sort_values("rmse").iloc[0]
        source_holdout_line = (
            f"best {best_row['source']} RMSE {float(best_row['rmse']):.3f}, "
            f"mean recall@20% {float(source_holdout_df['recall_top20pct'].mean()):.3f}"
        )
    rediscovery_line = "n/a"
    if rediscovery_summary:
        rediscovery_line = (
            f"protected top10 {float(rediscovery_summary.get('protected_top10_recall', 0.0)):.3f}, "
            f"naive top10 {float(rediscovery_summary.get('naive_top10_recall', 0.0)):.3f}, "
            f"protected median rank {float(rediscovery_summary.get('protected_median_positive_rank', 0.0)):.1f}"
        )
    gpu_scaffold_line = "n/a"
    gpu_scaffold_rows = [row for row in gpu_gnn_summary.get("splits", []) if row.get("split") == "scaffold"]
    if gpu_scaffold_rows:
        best_gpu = sorted(gpu_scaffold_rows, key=lambda row: float(row.get("rmse", 999.0)))[0]
        gpu_scaffold_line = f"{best_gpu.get('model', 'n/a')} | RMSE {float(best_gpu.get('rmse', 0.0)):.3f}"

    lines = [
        "# OncoForge ISEF Summary",
        "",
        "## Project Goal",
        "OncoForge is an AI-assisted lead-optimization pipeline for EGFR inhibitors.",
        "The system does not claim to discover finished drugs; it prioritizes chemically plausible, high-potential candidates for downstream wet-lab validation.",
        "",
        "## Upgraded Methodology",
        "- A multi-agent scorer now separates potency, chemistry, safety, novelty and applicability-domain checks instead of relying on a single scalar reward.",
        "- Cross-database validation now spans ChEMBL, BindingDB, IUPHAR, PubChem, Papyrus and ExCAPE-DB so candidate support is checked across independent public sources.",
        "- Verified reward is combined with anti-reward-hacking audits so suspicious molecules are penalized even if they exploit a proxy metric.",
        "- A verifiable-reward RL loop now optimizes traceable medicinal-chemistry actions instead of relying only on heuristic beam search.",
        "- The generator is now reaction-aware and scaffold-preserving, with hard constraints applied during generation rather than only after scoring.",
        "- GPU stages now benchmark a graph neural model and a neural DQN policy on the same candidate ecosystem as the classical pipeline.",
        "- Candidate feasibility is scored with non-experimental evidence: active-neighbor support, scaffold support, fragment support and generation traceability.",
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
        f"- Temporal split RMSE / R2: `{temporal_metrics.get('rmse', 'n/a')}` / `{temporal_metrics.get('r2', 'n/a')}`",
        "",
        "## Audit Diagnostics",
        f"- Audit pass rate: `{notebook_metrics.get('audit_pass_rate', 'n/a')}`",
        f"- Audit review rate: `{notebook_metrics.get('audit_review_rate', 'n/a')}`",
        f"- Audit fail rate: `{notebook_metrics.get('audit_fail_rate', 'n/a')}`",
        f"- Median reward hacking risk: `{notebook_metrics.get('median_reward_hacking_risk', 'n/a')}`",
        f"- Mean audit demotion: `{notebook_metrics.get('mean_audit_demotion', 'n/a')}` positions",
        f"- Feasibility pass rate on optimized candidates: `{feasibility_pass_rate}`",
        f"- Best Vina affinity on optimized candidates: `{best_vina_affinity}`",
        f"- Mean interaction support on optimized candidates: `{notebook_metrics.get('mean_interaction_support', 'n/a')}`",
        f"- Mean experimental readiness: `{notebook_metrics.get('mean_experimental_readiness', 'n/a')}`",
        f"- Cross-database mean consensus: `{notebook_metrics.get('cross_database_mean_consensus', 'n/a')}`",
        f"- Cross-database strong rate: `{notebook_metrics.get('cross_database_strong_rate', 'n/a')}`",
        f"- External evidence mean support: `{notebook_metrics.get('external_evidence_mean_support', 'n/a')}`",
        f"- External evidence pass rate: `{notebook_metrics.get('external_evidence_pass_rate', 'n/a')}`",
        f"- Evidence arbiter mean support: `{notebook_metrics.get('evidence_arbiter_mean_support', 'n/a')}`",
        f"- Evidence arbiter pass rate: `{notebook_metrics.get('evidence_arbiter_pass_rate', 'n/a')}`",
        f"- Papyrus molecules / mean support: `{papyrus_summary.get('n_unique_molecules', 'n/a')}` / `{papyrus_summary.get('mean_support_score', 'n/a')}`",
        f"- ExCAPE molecules / mean support: `{excape_summary.get('n_unique_molecules', 'n/a')}` / `{excape_summary.get('mean_support_score', 'n/a')}`",
        f"- PubChem mean enriched evidence: `{pubchem_summary.get('mean_enriched_evidence_score', 'n/a')}`",
        f"- PubChem strong evidence rate: `{pubchem_summary.get('strong_evidence_rate', 'n/a')}`",
        f"- PubChem virtual/proxy exposure rate: `{pubchem_summary.get('virtual_proxy_exposed_rate', 'n/a')}`",
        f"- Prospective validation batch size: `{notebook_metrics.get('prospective_batch_size', 'n/a')}`",
        f"- Prospective mean acquisition score: `{notebook_metrics.get('prospective_mean_acquisition_score', 'n/a')}`",
        f"- Prospective mean structure-evidence support: `{notebook_metrics.get('prospective_mean_structure_evidence_support', 'n/a')}`",
        f"- Broad analog benchmark count / mean generator priority: `{generated_summary.get('n_candidates', 'n/a')}` / `{generated_summary.get('mean_generator_priority_score', 'n/a')}`",
        f"- Broad analog mean adaptive prior: `{notebook_metrics.get('generated_mean_adaptive_action_prior', 'n/a')}`",
        f"- AI-guided benchmark count / mean generator priority: `{ai_guided_summary.get('n_candidates', 'n/a')}` / `{ai_guided_summary.get('mean_generator_priority_score', 'n/a')}`",
        f"- AI-guided mean adaptive prior: `{notebook_metrics.get('ai_guided_mean_adaptive_action_prior', 'n/a')}`",
        f"- Iterative benchmark count / top mean final score: `{iterative_summary.get('n_candidates', 'n/a')}` / `{iterative_summary.get('top_mean_final_score', 'n/a')}`",
        f"- Iterative mean adaptive prior: `{notebook_metrics.get('iterative_mean_adaptive_action_prior', 'n/a')}`",
        f"- Generator suite artifact present: `{generation_suite_df is not None and not generation_suite_df.empty}`",
        f"- Ablation suite artifact present: `{ablation_df is not None and not ablation_df.empty}`",
        f"- RL mean cross-database consensus: `{notebook_metrics.get('rl_mean_cross_database_consensus', 'n/a')}`",
        f"- RL mean external evidence support: `{notebook_metrics.get('rl_mean_external_evidence_support', 'n/a')}`",
        f"- RL mean structure-evidence support: `{notebook_metrics.get('rl_mean_structure_evidence_support', 'n/a')}`",
        f"- RL ready rate: `{notebook_metrics.get('rl_readiness_ready_rate', 'n/a')}`",
        f"- GPU GNN scaffold snapshot: `{gpu_scaffold_line}`",
        f"- GPU RL mean external evidence support: `{notebook_metrics.get('gpu_rl_mean_external_evidence_support', 'n/a')}`",
        f"- GPU RL mean structure-evidence support: `{notebook_metrics.get('gpu_rl_mean_structure_evidence_support', 'n/a')}`",
        f"- GPU RL best episode return: `{notebook_metrics.get('gpu_rl_best_episode_return', 'n/a')}`",
        f"- GPU actor-critic mean external evidence support: `{notebook_metrics.get('gpu_actor_critic_mean_external_evidence_support', 'n/a')}`",
        f"- GPU actor-critic mean structure-evidence support: `{notebook_metrics.get('gpu_actor_critic_mean_structure_evidence_support', 'n/a')}`",
        f"- GPU actor-critic best episode return: `{notebook_metrics.get('gpu_actor_critic_best_episode_return', actor_critic_summary.get('best_episode_return', 'n/a'))}`",
        f"- Best repeated-seed scaffold model: `{robust_line}`",
        f"- Reward-hacking challenge snapshot: `{challenge_line}`",
        f"- Source holdout snapshot: `{source_holdout_line}`",
        f"- Rediscovery benchmark snapshot: `{rediscovery_line}`",
        "",
        "## Verifiable RL Diagnostics",
        f"- Best episode return: `{rl_summary.get('best_episode_return', 'n/a')}`",
        f"- Mean episode return: `{rl_summary.get('mean_episode_return', 'n/a')}`",
        "",
        "## GPU RL Diagnostics",
        f"- Best episode return: `{gpu_rl_summary.get('best_episode_return', 'n/a')}`",
        f"- Mean episode return: `{gpu_rl_summary.get('mean_episode_return', 'n/a')}`",
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
            ["name", "predicted_pIC50", "vina_affinity_kcal", "interaction_support_score", "docking_rescore", "final_score"],
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
        "## Feasibility-Supported Optimized Candidates",
        _format_top_table(
            feasibility_df,
            ["smiles", "predicted_pIC50", "feasibility_score", "vina_affinity_kcal", "feasibility_status", "max_active_similarity"],
        ),
        "",
        "## Experimental-Readiness Snapshot",
        _format_top_table(
            readiness_df,
            [
                "smiles",
                "predicted_pIC50",
                "experimental_readiness_score",
                "experimental_readiness_status",
                "experimental_track",
                "docking_rescore",
            ],
        ),
        "",
        "## Cross-Database Validation Snapshot",
        _format_top_table(
            crossdb_df,
            [
                "smiles",
                "predicted_pIC50",
                "cross_database_consensus_score",
                "external_evidence_support",
                "cross_database_independent_support_count",
                "cross_database_status",
                "experimental_readiness_score",
            ],
        ),
        "",
        "## Prospective Validation Batch",
        _format_top_table(
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
        "",
        "## Verifiable RL Candidates",
        _format_top_table(
            rl_df,
            ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "experimental_readiness_score", "rl_priority_score"],
        ),
        "",
        "## GPU DQN RL Candidates",
        _format_top_table(
            gpu_rl_df,
            ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "evidence_arbiter_support", "gpu_rl_priority_score"],
        ),
        "",
        "## GPU Actor-Critic Candidates",
        _format_top_table(
            actor_critic_df,
            ["smiles", "predicted_pIC50", "cross_database_consensus_score", "external_evidence_support", "experimental_readiness_score", "actor_critic_priority_score"],
        ),
        "",
        "## Main Artifacts",
        "- `reports/model_performance_summary.json`",
        "- `reports/model_robustness_summary.csv`",
        "- `reports/gpu_gnn_benchmark.csv`",
        "- `reports/gpu_gnn_performance_summary.json`",
        "- `reports/ranked_egfr_dataset.csv`",
        "- `reports/marketed_egfr_scored.csv`",
        "- `reports/marketed_egfr_structural_benchmark.csv`",
        "- `reports/generated_analogs_ranked.csv`",
        "- `reports/generated_analogs_ranked.summary.json`",
        "- `reports/ai_guided_analogs.summary.json`",
        "- `reports/iterative_ai_optimized_candidates.summary.json`",
        "- `reports/generation_benchmark_suite.csv`",
        "- `reports/studii_ablatie/studii_ablatie.csv`",
        "- `reports/studii_ablatie/rezumat_studii_ablatie.md`",
        "- `reports/generated_analogs_ranked_structural_crossdb.csv`",
        "- `reports/ai_guided_analogs_structural_crossdb.csv`",
        "- `reports/iterative_ai_optimized_candidates.csv`",
        "- `reports/iterative_ai_optimized_candidates_structural_feasibility.csv`",
        "- `reports/iterative_ai_optimized_candidates_structural_crossdb.csv`",
        "- `reports/iterative_ai_optimized_candidates_structural_crossdb.summary.json`",
        "- `data/processed/pubchem_egfr_reference.csv`",
        "- `data/processed/papyrus_egfr_reference.csv`",
        "- `data/processed/papyrus_egfr_reference.summary.json`",
        "- `data/processed/excape_egfr_reference.csv`",
        "- `data/processed/excape_egfr_reference.summary.json`",
        "- `data/processed/pubchem_egfr_reference.summary.json`",
        "- `data/processed/pubchem_egfr_assay_catalog.csv`",
        "- `reports/final_diverse_candidates.csv`",
        "- `reports/market_comparable_novel_shortlist.csv`",
        "- `reports/prospective_validation_batch.csv`",
        "- `reports/prospective_validation_batch.summary.json`",
        "- `reports/rl_verifiable/rl_top_candidates.csv`",
        "- `reports/rl_verifiable/rl_top_candidates_crossdb.csv`",
        "- `reports/rl_verifiable/rl_training_summary.json`",
        "- `reports/rl_gpu_dqn/gpu_rl_top_candidates.csv`",
        "- `reports/rl_gpu_dqn/gpu_rl_training_summary.json`",
        "- `reports/rl_gpu_actor_critic/gpu_actor_critic_top_candidates.csv`",
        "- `reports/rl_gpu_actor_critic/gpu_actor_critic_summary.json`",
        "- `reports/reward_hacking_challenge/reward_hacking_challenge_summary.csv`",
        "- `reports/source_holdout_benchmark.csv`",
        "- `reports/source_holdout_benchmark.json`",
        "- `reports/rediscovery_benchmark/rediscovery_panel.csv`",
        "- `reports/rediscovery_benchmark/rediscovery_summary.json`",
        "- `reports/technical_notebook/technical_notebook_summary.md`",
        "- `reports/technical_notebook/technical_notebook_metrics.json`",
        "- `reports/technical_notebook_history/context_memory.md`",
        "- `reports/technical_notebook_quick/technical_notebook_summary.md`",
        "- `reports/technical_notebook_quick/technical_notebook_metrics.json`",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved project summary: {summary_path}")


if __name__ == "__main__":
    main()
