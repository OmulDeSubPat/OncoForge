from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT


def main():
    in_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing file: {in_path}\n"
            "Run: python -m src.generation.iterative_ai_optimizer"
        )

    df = pd.read_csv(in_path)

    # Summary by round
    round_summary = (
        df.groupby("round")
        .agg(
            n_candidates=("smiles", "count"),
            avg_score=("final_score", "mean"),
            max_score=("final_score", "max"),
            avg_pIC50=("predicted_pIC50", "mean"),
            avg_qed=("QED", "mean"),
            avg_uncertainty=("uncertainty", "mean"),
        )
        .reset_index()
        .sort_values("round")
    )

    # Best candidate per round
    idx = df.groupby("round")["final_score"].idxmax()
    best_per_round = df.loc[idx].sort_values("round").reset_index(drop=True)

    # Best parent seeds overall
    parent_summary = (
        df.groupby("parent_seed")
        .agg(
            n_generated=("smiles", "count"),
            best_score=("final_score", "max"),
            avg_score=("final_score", "mean"),
        )
        .reset_index()
        .sort_values("best_score", ascending=False)
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    round_summary.to_csv(reports_dir / "optimization_round_summary.csv", index=False)
    best_per_round.to_csv(reports_dir / "best_candidate_per_round.csv", index=False)
    parent_summary.to_csv(reports_dir / "best_parent_seeds.csv", index=False)

    print("[OK] Saved:")
    print(" - reports/optimization_round_summary.csv")
    print(" - reports/best_candidate_per_round.csv")
    print(" - reports/best_parent_seeds.csv")

    print("\n=== Round summary ===")
    print(round_summary.to_string(index=False))

    print("\n=== Best candidate per round ===")
    print(
        best_per_round[
            ["round", "smiles", "predicted_pIC50", "QED", "uncertainty", "final_score", "parent_seed"]
        ].to_string(index=False)
    )

    print("\n=== Top parent seeds ===")
    print(parent_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()