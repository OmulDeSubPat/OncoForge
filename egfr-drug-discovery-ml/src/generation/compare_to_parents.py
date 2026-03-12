from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT


def main():
    generated_path = PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"

    if not generated_path.exists():
        raise FileNotFoundError(
            f"Missing file: {generated_path}\n"
            "Run: python -m src.generation.generate_and_rank_analogs"
        )

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing file: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    gen = pd.read_csv(generated_path)
    ranked = pd.read_csv(ranked_path)

    parent_cols = [
        "smiles",
        "predicted_pIC50",
        "uncertainty",
        "QED",
        "final_score",
    ]

    parents = ranked[parent_cols].copy()
    parents = parents.rename(columns={
        "smiles": "parent_seed",
        "predicted_pIC50": "parent_predicted_pIC50",
        "uncertainty": "parent_uncertainty",
        "QED": "parent_QED",
        "final_score": "parent_final_score",
    })

    merged = gen.merge(parents, on="parent_seed", how="left")

    merged["delta_predicted_pIC50"] = (
        merged["predicted_pIC50"] - merged["parent_predicted_pIC50"]
    )
    merged["delta_QED"] = merged["QED"] - merged["parent_QED"]
    merged["delta_final_score"] = merged["final_score"] - merged["parent_final_score"]

    merged = merged.sort_values("delta_final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "analogs_vs_parents.csv"
    merged.to_csv(out_path, index=False)

    print(f"[OK] Saved comparison file: {out_path}")
    print(
        merged[
            [
                "smiles",
                "parent_seed",
                "predicted_pIC50",
                "parent_predicted_pIC50",
                "delta_predicted_pIC50",
                "QED",
                "parent_QED",
                "delta_QED",
                "final_score",
                "parent_final_score",
                "delta_final_score",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()