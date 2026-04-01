# OncoForge

OncoForge is an AI-assisted EGFR lead-optimization project focused on **prioritizing** candidate molecules, not claiming in-silico drug discovery as a finished endpoint.

## What Makes This Version Stronger

- Multi-agent scoring:
  - `potency agent` uses a Random Forest ensemble with uncertainty
  - `chemistry agent` scores QED, SA, descriptor sanity and Lipinski pressure
  - `safety agent` checks PAINS and structural alerts
  - `novelty/applicability agent` balances novelty against the known EGFR chemical space
  - `audit agent` flags reward-hacking patterns
- Reward design:
  - verified reward instead of a single naive score
  - veto logic for clearly problematic molecules
  - explicit `reward_hacking_risk` and `reward_hacking_flags`
- ML rigor:
  - ensemble evaluation on random and scaffold splits
  - final ensemble retrained on the full dataset after evaluation
- Candidate quality:
  - market comparison
  - diversity filtering
  - shortlist generation for market-comparable but still novel molecules

## Core Idea

The project addresses a real problem in computational drug discovery: the search space is enormous, while wet-lab validation is expensive. The point of the system is to reduce the search space intelligently and deliver stronger candidates to experimental follow-up.

## Main Pipeline

1. `python -m src.models.train_multiview_ensemble`
2. `python -m src.models.rank_dataset`
3. `python -m src.structure.dock_marketed_egfr`
4. `python -m src.generation.generate_and_rank_analogs`
5. `python -m src.generation.generate_ai_guided_analogs`
6. `python -m src.generation.iterative_ai_optimizer`
7. `python -m src.feasibility.assess_candidates`
8. `python -m src.benchmark.compare_candidates_to_market`
9. `python -m src.benchmark.select_market_comparable_novel`
10. `python -m src.generation.select_diverse_candidates`
11. `python -m src.models.run_model_robustness_benchmark`
12. `python -m src.evaluation.run_source_holdout_benchmark`
13. `python -m src.evaluation.run_reward_hacking_challenge`
14. `python -m src.evaluation.run_rediscovery_benchmark`
15. `python -m src.rl.train_verifiable_rl`
16. `python -m src.pipelines.build_project_summary`

Or run the full upgraded flow with:

```bash
python -m src.pipelines.run_isef_pipeline
```

Useful options:

```bash
python -m src.pipelines.run_isef_pipeline --skip-training
python -m src.pipelines.run_isef_pipeline --refresh-clean
python -m src.pipelines.run_isef_pipeline --summary-only
python -m src.pipelines.run_isef_pipeline --glossary-only
```

## Key Outputs

- `reports/model_performance_summary.json`
- `reports/model_robustness_summary.csv`
- `reports/ranked_egfr_dataset.csv`
- `reports/marketed_egfr_structural_benchmark.csv`
- `reports/generated_analogs_ranked.csv`
- `reports/iterative_ai_optimized_candidates.csv`
- `reports/iterative_ai_optimized_candidates_feasibility.csv`
- `reports/final_diverse_candidates.csv`
- `reports/market_comparable_novel_shortlist.csv`
- `reports/reward_hacking_challenge/reward_hacking_challenge_summary.csv`
- `reports/source_holdout_benchmark.csv`
- `reports/rediscovery_benchmark/rediscovery_summary.json`
- `reports/isef_project_summary.md`
- `reports/OncoForge_Technical_Notebook.docx`
- `reports/OncoForge_Buzzword_Glossary.docx`

## Reproducibility

- `python -m src.pipelines.bootstrap_reproducibility` initializes the standard history and benchmark skeletons in the project root.
- `python -m src.pipelines.bootstrap_reproducibility --check-only` verifies that the expected reproducibility files are present.
- `python -m unittest tests.test_reproducibility -v` checks the reproducibility helpers without requiring the full chemistry stack.

Standard root artifacts are:

- `valori_R2.csv`
- `valori_RMSE.csv`
- `valori_MAE.csv`
- `valori_MSE.csv`
- `valori_pIC50.csv`
- `valori_IC50.csv`
- `valori_Pearson.csv`
- `valori_Spearman.csv`
- `valori_Incertitudine.csv`
- `istoric_metrici.csv`
- `benchmark_studii.csv`
- `comparatii_literatura.csv`

## Project Structure

- `src/data/` data ingestion, cleaning and source merging
- `src/features/` featurization and descriptors
- `src/models/` QSAR training and robustness checks
- `src/generation/` molecule generation, ranking and lineage tracking
- `src/feasibility/` synthetic feasibility and readiness scoring
- `src/evaluation/` validation, ablation and benchmark scripts
- `src/structure/` docking and structural rescoring
- `src/rl/` verifiable RL and GPU RL experiments
- `src/pipelines/` orchestration, summaries and reproducibility helpers
- `reports/` generated analysis artifacts

## Project Positioning

This project sits at the intersection of:

- cheminformatics
- computer-aided drug discovery
- machine learning for molecular design
- multi-objective optimization
- safe reward design / anti-reward-hacking

That framing is strong for ISEF because the novelty is not just “AI generates molecules”, but **how the optimization is orchestrated, audited, and validated**.
