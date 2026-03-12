from __future__ import annotations

import pandas as pd
from src.config import PROJECT_ROOT


def main():
    analogs_path = PROJECT_ROOT / "reports" / "ai_guided_analogs.csv"
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"

    if not analogs_path.exists():
        raise FileNotFoundError(
            f"Missing analog file: {analogs_path}\n"
            "Run: python -m src.generation.generate_ai_guided_analogs"
        )

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    analogs = pd.read_csv(analogs_path)
    ranked = pd.read_csv(ranked_path)

    parents = ranked[["smiles", "predicted_pIC50", "QED", "final_score"]].copy()
    parents = parents.rename(columns={
        "smiles": "parent_seed",
        "predicted_pIC50": "parent_predicted_pIC50",
        "QED": "parent_QED",
        "final_score": "parent_final_score",
    })

    merged = analogs.merge(parents, on="parent_seed", how="left")

    merged["delta_predicted_pIC50"] = merged["predicted_pIC50"] - merged["parent_predicted_pIC50"]
    merged["delta_QED"] = merged["QED"] - merged["parent_QED"]
    merged["delta_final_score"] = merged["final_score"] - merged["parent_final_score"]

    merged = merged.sort_values("delta_final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "ai_guided_analogs_vs_parents.csv"
    merged.to_csv(out_path, index=False)

    print(f"[OK] Saved comparison: {out_path}")
    print(
        merged[
            [
                "smiles",
                "parent_seed",
                "delta_predicted_pIC50",
                "delta_QED",
                "delta_final_score",
            ]
        ].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main()