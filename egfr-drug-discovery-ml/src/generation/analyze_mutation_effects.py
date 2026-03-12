from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT


def classify_delta(x: float) -> str:
    if x >= 0.10:
        return "improved"
    elif x <= -0.10:
        return "worse"
    return "neutral"


def main():
    in_path = PROJECT_ROOT / "reports" / "analogs_vs_parents.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing file: {in_path}\n"
            "Run: python -m src.generation.compare_to_parents"
        )

    df = pd.read_csv(in_path)

    df["effect_class"] = df["delta_final_score"].apply(classify_delta)

    summary = (
        df.groupby("effect_class")
        .agg(
            n=("smiles", "count"),
            avg_delta_score=("delta_final_score", "mean"),
            avg_delta_pIC50=("delta_predicted_pIC50", "mean"),
            avg_delta_QED=("delta_QED", "mean"),
        )
        .reset_index()
    )

    out_path = PROJECT_ROOT / "reports" / "mutation_effect_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"[OK] Saved summary: {out_path}")
    print(summary.to_string(index=False))

    print("\nTop 10 best analogs by delta_final_score:")
    print(
        df.sort_values("delta_final_score", ascending=False)[
            [
                "smiles",
                "parent_seed",
                "delta_predicted_pIC50",
                "delta_QED",
                "delta_final_score",
                "effect_class",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTop 10 worst analogs by delta_final_score:")
    print(
        df.sort_values("delta_final_score", ascending=True)[
            [
                "smiles",
                "parent_seed",
                "delta_predicted_pIC50",
                "delta_QED",
                "delta_final_score",
                "effect_class",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()