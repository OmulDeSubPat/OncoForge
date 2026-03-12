from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT


def main():
    reports_dir = PROJECT_ROOT / "reports"

    ranked_path = reports_dir / "ranked_egfr_dataset.csv"
    round_summary_path = reports_dir / "optimization_round_summary.csv"

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing file: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    ranked = pd.read_csv(ranked_path)

    # 1. Histogram predicted_pIC50
    plt.figure(figsize=(8, 5))
    plt.hist(ranked["predicted_pIC50"], bins=40)
    plt.xlabel("Predicted pIC50")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted EGFR Potency")
    plt.tight_layout()
    plt.savefig(reports_dir / "hist_predicted_pIC50.png", dpi=300)
    plt.close()

    # 2. Histogram QED
    plt.figure(figsize=(8, 5))
    plt.hist(ranked["QED"], bins=40)
    plt.xlabel("QED")
    plt.ylabel("Count")
    plt.title("Distribution of Drug-Likeness (QED)")
    plt.tight_layout()
    plt.savefig(reports_dir / "hist_qed.png", dpi=300)
    plt.close()

    # 3. Scatter predicted_pIC50 vs QED
    plt.figure(figsize=(8, 6))
    plt.scatter(ranked["QED"], ranked["predicted_pIC50"], alpha=0.5)
    plt.xlabel("QED")
    plt.ylabel("Predicted pIC50")
    plt.title("Potency vs Drug-Likeness")
    plt.tight_layout()
    plt.savefig(reports_dir / "scatter_pIC50_vs_QED.png", dpi=300)
    plt.close()

    # 4. Optimization trajectory, dacă există
    if round_summary_path.exists():
        round_summary = pd.read_csv(round_summary_path)

        plt.figure(figsize=(8, 5))
        plt.plot(round_summary["round"], round_summary["avg_score"], marker="o")
        plt.xlabel("Optimization Round")
        plt.ylabel("Average Final Score")
        plt.title("Average Candidate Score Across Optimization Rounds")
        plt.tight_layout()
        plt.savefig(reports_dir / "trajectory_avg_score.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(round_summary["round"], round_summary["max_score"], marker="o")
        plt.xlabel("Optimization Round")
        plt.ylabel("Best Final Score")
        plt.title("Best Candidate Score Across Optimization Rounds")
        plt.tight_layout()
        plt.savefig(reports_dir / "trajectory_best_score.png", dpi=300)
        plt.close()

    print("[OK] Saved report plots in reports/")


if __name__ == "__main__":
    main()