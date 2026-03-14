from __future__ import annotations

import argparse

from src.benchmark.compare_candidates_to_market import main as compare_candidates_to_market_main
from src.benchmark.score_marketed_egfr import main as score_marketed_egfr_main
from src.benchmark.select_market_comparable_novel import main as select_market_comparable_novel_main
from src.config import PROJECT_ROOT
from src.data.clean_egfr_ic50 import main as clean_data_main
from src.generation.analyze_optimization_trajectory import main as analyze_optimization_trajectory_main
from src.generation.generate_ai_guided_analogs import main as generate_ai_guided_analogs_main
from src.generation.generate_and_rank_analogs import main as generate_and_rank_analogs_main
from src.generation.iterative_ai_optimizer import main as iterative_ai_optimizer_main
from src.generation.select_diverse_candidates import main as select_diverse_candidates_main
from src.models.rank_dataset import main as rank_dataset_main
from src.models.train_multiview_ensemble import main as train_multiview_ensemble_main
from src.pipelines.artifact_utils import load_csv_artifact
from src.pipelines.build_project_summary import main as build_project_summary_main
from src.pipelines.build_technical_notebook_assets import main as build_technical_notebook_assets_main
from src.structure.rescore_top_candidates import main as rescore_top_candidates_main


def _stage(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    fn()


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
    args = parser.parse_args()

    processed_path = PROJECT_ROOT / "data" / "processed" / "egfr_chembl_ic50_clean.csv"

    if args.summary_only:
        _stage("Build Summary", build_project_summary_main)
        return

    if args.notebook_only:
        _validate_ranked_dataset_if_present()
        _stage("Build Technical Notebook Assets", lambda: build_technical_notebook_assets_main([]))
        return

    if args.refresh_clean or not processed_path.exists():
        _stage("Clean Data", clean_data_main)

    if not args.skip_training:
        _stage("Train Ensemble", train_multiview_ensemble_main)

    _stage("Rank Dataset", lambda: rank_dataset_main([]))
    _validate_ranked_dataset_if_present()
    _stage("Score Marketed Drugs", score_marketed_egfr_main)
    _stage("Generate String Analogs", generate_and_rank_analogs_main)
    _stage(
        "Structural Rescore Generated Analogs",
        lambda: rescore_top_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv"),
                "--top-k",
                "250",
            ]
        ),
    )
    _stage("Generate AI-Guided Analogs", generate_ai_guided_analogs_main)
    _stage("Iterative Optimizer", iterative_ai_optimizer_main)
    _stage(
        "Structural Rescore Optimized Candidates",
        lambda: rescore_top_candidates_main(
            [
                "--input",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"),
                "--out",
                str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
                "--top-k",
                "250",
            ]
        ),
    )
    _stage("Compare To Market", compare_candidates_to_market_main)
    _stage("Select Market-Comparable Novel Candidates", select_market_comparable_novel_main)
    _stage("Select Diverse Candidates", select_diverse_candidates_main)
    _stage("Analyze Optimization Trajectory", analyze_optimization_trajectory_main)
    _stage("Build Summary", build_project_summary_main)
    _stage("Build Technical Notebook Assets", lambda: build_technical_notebook_assets_main([]))


if __name__ == "__main__":
    main()
