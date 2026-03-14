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

1. `python -m src.models.train_qsar_rf_ensemble`
2. `python -m src.models.rank_dataset`
3. `python -m src.benchmark.score_marketed_egfr`
4. `python -m src.generation.generate_and_rank_analogs`
5. `python -m src.generation.generate_ai_guided_analogs`
6. `python -m src.generation.iterative_ai_optimizer`
7. `python -m src.benchmark.compare_candidates_to_market`
8. `python -m src.benchmark.select_market_comparable_novel`
9. `python -m src.generation.select_diverse_candidates`
10. `python -m src.pipelines.build_project_summary`

Or run the full upgraded flow with:

```bash
python -m src.pipelines.run_isef_pipeline
```

Useful options:

```bash
python -m src.pipelines.run_isef_pipeline --skip-training
python -m src.pipelines.run_isef_pipeline --refresh-clean
python -m src.pipelines.run_isef_pipeline --summary-only
```

## Key Outputs

- `reports/model_performance_summary.json`
- `reports/ranked_egfr_dataset.csv`
- `reports/marketed_egfr_scored.csv`
- `reports/generated_analogs_ranked.csv`
- `reports/iterative_ai_optimized_candidates.csv`
- `reports/final_diverse_candidates.csv`
- `reports/market_comparable_novel_shortlist.csv`
- `reports/isef_project_summary.md`

## Project Positioning

This project sits at the intersection of:

- cheminformatics
- computer-aided drug discovery
- machine learning for molecular design
- multi-objective optimization
- safe reward design / anti-reward-hacking

That framing is strong for ISEF because the novelty is not just “AI generates molecules”, but **how the optimization is orchestrated, audited, and validated**.
