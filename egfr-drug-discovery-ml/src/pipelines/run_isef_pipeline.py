from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.benchmark.compare_candidates_to_market import main as compare_candidates_to_market_main
from src.benchmark.score_marketed_egfr import main as score_marketed_egfr_main
from src.benchmark.select_market_comparable_novel import main as select_market_comparable_novel_main
from src.config import PROJECT_ROOT
from src.data.clean_egfr_ic50 import main as clean_data_main
from src.data.fetch_excape_egfr import main as fetch_excape_egfr_main
from src.data.fetch_iuphar_egfr import main as fetch_iuphar_egfr_main
from src.data.fetch_papyrus_egfr import main as fetch_papyrus_egfr_main
from src.data.fetch_pubchem_egfr import main as fetch_pubchem_egfr_main
from src.data.merge_egfr_sources import main as merge_egfr_sources_main
from src.evaluation.run_cross_database_validation import main as run_cross_database_validation_main
from src.evaluation.run_rediscovery_benchmark import main as run_rediscovery_benchmark_main
from src.feasibility.assess_candidates import main as assess_candidates_main
from src.feasibility.score_experimental_readiness import main as score_experimental_readiness_main
from src.generation.analyze_optimization_trajectory import main as analyze_optimization_trajectory_main
from src.generation.generate_ai_guided_analogs import main as generate_ai_guided_analogs_main
from src.generation.generate_and_rank_analogs import main as generate_and_rank_analogs_main
from src.generation.iterative_ai_optimizer import main as iterative_ai_optimizer_main
from src.generation.select_diverse_candidates import main as select_diverse_candidates_main
from src.models.rank_dataset import main as rank_dataset_main
from src.models.run_model_robustness_benchmark import main as run_model_robustness_benchmark_main
from src.evaluation.run_source_holdout_benchmark import main as run_source_holdout_benchmark_main
from src.models.train_multiview_ensemble import main as train_multiview_ensemble_main
from src.pipelines.artifact_utils import load_csv_artifact
from src.pipelines.build_technical_notebook_assets import main as build_technical_notebook_assets_main
from src.pipelines.build_buzzword_glossary_docx import main as build_buzzword_glossary_docx_main
from src.pipelines.build_project_summary import main as build_project_summary_main
from src.pipelines.build_technical_notebook_docx import main as build_technical_notebook_docx_main
from src.evaluation.run_reward_hacking_challenge import main as run_reward_hacking_challenge_main
from src.evaluation.select_prospective_validation_batch import main as select_prospective_validation_batch_main
from src.rl.train_verifiable_rl import main as train_verifiable_rl_main
from src.structure.annotate_structural_interactions import main as annotate_structural_interactions_main
from src.structure.dock_marketed_egfr import main as dock_marketed_egfr_main
from src.structure.rescore_top_candidates import main as rescore_top_candidates_main


def _stage(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    fn()


def _external_python_path(candidate: str | None = None) -> Path:
    if candidate:
        return Path(candidate)
    preferred = [
        PROJECT_ROOT / ".venv312-gpu" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for path in preferred:
        if path.exists():
            return path
    return Path(sys.executable)


def _run_external_stage(name: str, module: str, args: list[str], python_path: Path) -> None:
    print(f"\n=== {name} ===")
    command = [str(python_path), "-m", module, *args]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _validate_ranked_dataset_if_present() -> None:
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    if not ranked_path.exists():
        return
    load_csv_artifact(
        ranked_path,
        required_columns=[
            "smiles",
            "predicted_pIC50",
            "QED",
            "reward_hacking_risk",
            "agent_disagreement_score",
            "audit_status",
            "applicability_score",
            "veto",
            "final_score",
        ],
        producer="python -m src.models.rank_dataset",
    )


def main():
    parser = argparse.ArgumentParser(description="Run the upgraded OncoForge pipeline.")
    parser.add_argument(
        "--refresh-clean",
        action="store_true",
        help="Rebuild the processed dataset from the raw ChEMBL export.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse the existing ensemble model instead of retraining it.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild the markdown summary from existing artifacts.",
    )
    parser.add_argument(
        "--notebook-only",
        action="store_true",
        help="Only rebuild the technical notebook plots and markdown assets.",
    )
    parser.add_argument(
        "--glossary-only",
        action="store_true",
        help="Only rebuild the buzzword glossary Word document.",
    )
    parser.add_argument(
        "--structural-top-k",
        type=int,
        default=60,
        help="How many top candidates to send through the structural docking stage for each major generated set.",
    )
    parser.add_argument(
        "--structural-backend",
        type=str,
        choices=["auto", "reference", "vina"],
        default="auto",
        help="Structural backend for candidate rescoring. 'auto' prefers Vina and falls back to reference-ligand support.",
    )
    parser.add_argument(
        "--vina-exhaustiveness",
        type=int,
        default=6,
        help="AutoDock Vina exhaustiveness used during structural rescoring.",
    )
    parser.add_argument(
        "--vina-cpu",
        type=int,
        default=1,
        help="CPU count passed to AutoDock Vina for each docking job.",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip verifiable RL training and keep the most recent RL artifacts.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip repeated-seed robustness benchmarking.",
    )
    parser.add_argument(
        "--skip-source-holdout",
        action="store_true",
        help="Skip the leave-one-source-out benchmark.",
    )
    parser.add_argument(
        "--skip-rediscovery",
        action="store_true",
        help="Skip the rediscovery benchmark against known EGFR positives.",
    )
    parser.add_argument(
        "--skip-gpu-models",
        action="store_true",
        help="Skip the separate GPU environment stages for the graph model and neural RL.",
    )
    parser.add_argument(
        "--gpu-python",
        type=str,
        default=None,
        help="Optional path to the Python executable used for GPU stages.",
    )
    parser.add_argument(
        "--gpu-gnn-epochs",
        type=int,
        default=24,
        help="Epoch count for the GPU graph model benchmark.",
    )
    parser.add_argument(
        "--gpu-rl-episodes",
        type=int,
        default=240,
        help="Episode count for the GPU DQN stage.",
    )
    parser.add_argument(
        "--gpu-rl-max-actions-per-family",
        type=int,
        default=3,
        help="How many top grounded variants the GPU DQN keeps per medicinal-chemistry family.",
    )
    parser.add_argument(
        "--gpu-rl-max-actions-total",
        type=int,
        default=24,
        help="Total grounded action budget exposed to the GPU DQN at each step.",
    )
    parser.add_argument(
        "--robustness-seeds",
        type=int,
        nargs="+",
        default=[11, 42, 93],
        help="Seeds used for repeated-seed robustness benchmarking.",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=72,
        help="Episode count for verifiable RL training.",
    )
    parser.add_argument(
        "--rl-max-steps",
        type=int,
        default=4,
        help="Maximum medicinal-chemistry steps per RL episode.",
    )
    parser.add_argument(
        "--rl-seed-pool-size",
        type=int,
        default=24,
        help="How many high-confidence ranked molecules seed RL training.",
    )
    parser.add_argument(
        "--rl-evaluation-rollouts",
        type=int,
        default=10,
        help="How many greedy seed rollouts to evaluate after RL training.",
    )
    parser.add_argument(
        "--rl-max-actions-per-family",
        type=int,
        default=3,
        help="How many top grounded variants tabular RL keeps per medicinal-chemistry family.",
    )
    parser.add_argument(
        "--rl-max-actions-total",
        type=int,
        default=24,
        help="Total grounded action budget exposed to tabular RL at each step.",
    )
    parser.add_argument("--analog-seed-count", type=int, default=24, help="Ranked seed count for the broad analog generation stage.")
    parser.add_argument("--analog-variants-per-seed", type=int, default=60, help="Variant count per seed for the broad analog generation stage.")
    parser.add_argument("--ai-guided-seed-count", type=int, default=15, help="Ranked seed count for the AI-guided analog stage.")
    parser.add_argument("--ai-guided-variants-per-seed", type=int, default=120, help="Variant count per seed for the AI-guided analog stage.")
    parser.add_argument("--iterative-seed-count", type=int, default=10, help="Initial seed count for iterative optimization.")
    parser.add_argument("--iterative-rounds", type=int, default=4, help="Number of iterative optimization rounds.")
    parser.add_argument("--iterative-beam-width", type=int, default=12, help="Beam width for iterative optimization.")
    parser.add_argument("--iterative-variants-per-seed", type=int, default=80, help="Variant count per seed during iterative optimization.")
    args = parser.parse_args()
    gpu_python = _external_python_path(args.gpu_python)

    processed_path = PROJECT_ROOT / "data" / "processed" / "egfr_chembl_ic50_clean.csv"

    if args.summary_only:
        _stage("Build Summary", build_project_summary_main)
        return

    if args.notebook_only:
        _validate_ranked_dataset_if_present()
        _stage("Build Technical Notebook Assets", lambda: build_technical_notebook_assets_main([]))
        _stage("Build Technical Notebook Word", build_technical_notebook_docx_main)
        return

    if args.glossary_only:
        _stage("Build Buzzword Glossary Word", build_buzzword_glossary_docx_main)
        return

    if args.refresh_clean or not processed_path.exists():
        _stage("Clean Data", clean_data_main)
    _stage("Fetch Papyrus EGFR Reference", fetch_papyrus_egfr_main)
    _stage("Fetch ExCAPE EGFR Reference", fetch_excape_egfr_main)
    _stage("Merge Multi-Source EGFR Dataset", merge_egfr_sources_main)

    if not args.skip_training:
        _stage("Train Ensemble", train_multiview_ensemble_main)

    _stage("Rank Dataset", lambda: rank_dataset_main([]))
    _validate_ranked_dataset_if_present()
    _stage("Score Marketed Drugs", score_marketed_egfr_main)
    _stage("Dock Marketed Drugs", dock_marketed_egfr_main)
    _stage("Fetch IUPHAR EGFR Reference", fetch_iuphar_egfr_main)
    _stage("Fetch PubChem EGFR Reference", fetch_pubchem_egfr_main)
    _stage(
        "Generate String Analogs",
        lambda: generate_and_rank_analogs_main(
            [
                "--seed-count",
                str(args.analog_seed_count),
                "--variants-per-seed",
                str(args.analog_variants_per_seed),
            ]
        ),
    )
    _stage(
        "Structural Rescore Generated Analogs",
        lambda: rescore_top_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv"),
                "--top-k",
                str(args.structural_top_k),
                "--backend",
                args.structural_backend,
                "--exhaustiveness",
                str(args.vina_exhaustiveness),
                "--cpu",
                str(args.vina_cpu),
                "--pose-dir",
                str(PROJECT_ROOT / "reports" / "vina_poses" / "generated_analogs"),
            ]
        ),
    )
    _stage(
        "Annotate Generated Analog Interactions",
        lambda: annotate_structural_interactions_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv"),
            ]
        ),
    )
    _stage(
        "Assess Generated Analog Feasibility",
        lambda: assess_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_feasibility.csv"),
                "--top-k",
                "250",
            ]
        ),
    )
    _stage(
        "Run Cross-Database Validation On Generated Analogs",
        lambda: run_cross_database_validation_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_feasibility.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_crossdb.csv"),
            ]
        ),
    )
    _stage(
        "Generate AI-Guided Analogs",
        lambda: generate_ai_guided_analogs_main(
            [
                "--seed-count",
                str(args.ai_guided_seed_count),
                "--variants-per-seed",
                str(args.ai_guided_variants_per_seed),
            ]
        ),
    )
    _stage(
        "Iterative Optimizer",
        lambda: iterative_ai_optimizer_main(
            [
                "--seed-count",
                str(args.iterative_seed_count),
                "--rounds",
                str(args.iterative_rounds),
                "--beam-width",
                str(args.iterative_beam_width),
                "--variants-per-seed",
                str(args.iterative_variants_per_seed),
            ]
        ),
    )
    _stage(
        "Structural Rescore Optimized Candidates",
        lambda: rescore_top_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
                "--top-k",
                str(args.structural_top_k),
                "--backend",
                args.structural_backend,
                "--exhaustiveness",
                str(args.vina_exhaustiveness),
                "--cpu",
                str(args.vina_cpu),
                "--pose-dir",
                str(PROJECT_ROOT / "reports" / "vina_poses" / "optimized_candidates"),
            ]
        ),
    )
    _stage(
        "Annotate Optimized Candidate Interactions",
        lambda: annotate_structural_interactions_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
            ]
        ),
    )
    _stage(
        "Assess Optimized Candidate Feasibility",
        lambda: assess_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv"),
                "--top-k",
                "250",
            ]
        ),
    )
    _stage(
        "Score Experimental Readiness",
        lambda: score_experimental_readiness_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv"),
            ]
        ),
    )
    _stage(
        "Run Cross-Database Validation",
        lambda: run_cross_database_validation_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv"),
            ]
        ),
    )
    _stage("Compare To Market", compare_candidates_to_market_main)
    _stage("Select Market-Comparable Novel Candidates", select_market_comparable_novel_main)
    _stage("Select Diverse Candidates", select_diverse_candidates_main)
    _stage(
        "Build Prospective Validation Batch",
        lambda: select_prospective_validation_batch_main(
            [
                "--batch-size",
                "18",
                "--out",
                str(PROJECT_ROOT / "reports" / "prospective_validation_batch.csv"),
            ]
        ),
    )
    if not args.skip_robustness:
        _stage(
            "Run Model Robustness Benchmark",
            lambda: run_model_robustness_benchmark_main(
                ["--seeds", *[str(seed) for seed in args.robustness_seeds]]
            ),
        )
    if not args.skip_source_holdout:
        _stage("Run Source Holdout Benchmark", run_source_holdout_benchmark_main)
    if not args.skip_gpu_models and gpu_python.exists():
        _run_external_stage(
            "Train GPU Graph Model",
            "src.models.train_gpu_gnn",
            ["--epochs", str(args.gpu_gnn_epochs)],
            gpu_python,
        )
    _stage("Run Reward-Hacking Challenge", run_reward_hacking_challenge_main)
    if not args.skip_rediscovery:
        _stage("Run Rediscovery Benchmark", run_rediscovery_benchmark_main)
    if not args.skip_rl:
        _stage(
            "Train Verifiable RL",
            lambda: train_verifiable_rl_main(
                [
                    "--episodes",
                    str(args.rl_episodes),
                    "--max-steps",
                    str(args.rl_max_steps),
                    "--seed-pool-size",
                    str(args.rl_seed_pool_size),
                    "--evaluation-rollouts",
                    str(args.rl_evaluation_rollouts),
                    "--max-actions-per-family",
                    str(args.rl_max_actions_per_family),
                    "--max-actions-total",
                    str(args.rl_max_actions_total),
                ]
            ),
        )
    if not args.skip_gpu_models and gpu_python.exists():
        _run_external_stage(
            "Train GPU DQN RL",
            "src.rl.train_gpu_dqn",
            [
                "--episodes",
                str(args.gpu_rl_episodes),
                "--max-actions-per-family",
                str(args.gpu_rl_max_actions_per_family),
                "--max-actions-total",
                str(args.gpu_rl_max_actions_total),
            ],
            gpu_python,
        )
    _stage("Analyze Optimization Trajectory", analyze_optimization_trajectory_main)
    _stage("Build Summary", build_project_summary_main)
    _stage("Build Technical Notebook Assets", lambda: build_technical_notebook_assets_main([]))
    _stage("Build Technical Notebook Word", build_technical_notebook_docx_main)
    _stage("Build Buzzword Glossary Word", build_buzzword_glossary_docx_main)


if __name__ == "__main__":
    main()
