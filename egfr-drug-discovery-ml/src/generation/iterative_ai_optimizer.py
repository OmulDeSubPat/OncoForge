from __future__ import annotations

import joblib
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule
from src.generation.rgroup_generator import generate_rgroup_variants


def select_seed_pool(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    df = df.copy()
    df = df[
        (df["predicted_pIC50"] >= 8.8) &
        (df["QED"] >= 0.45) &
        (df["uncertainty"] <= 0.05) &
        (df["penalty"] <= 0.3)
    ]
    return df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def main():
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )

    ranked_df = pd.read_csv(ranked_path)
    models = joblib.load(model_path)

    # initial seeds
    current_pool = select_seed_pool(ranked_df, top_k=10)
    all_generated = []
    seen = set(current_pool["smiles"].tolist())

    n_rounds = 3
    beam_width = 10
    variants_per_seed = 60

    print(f"[INFO] Starting iterative optimization with {len(current_pool)} seed molecules")

    for round_idx in range(1, n_rounds + 1):
        print(f"\n[INFO] Round {round_idx}")

        candidates = []

        for _, row in tqdm(current_pool.iterrows(), total=len(current_pool), desc=f"Round {round_idx} generation"):
            parent_smiles = row["smiles"]
            variants = generate_rgroup_variants(parent_smiles, max_variants=variants_per_seed)

            for smi in variants:
                if smi in seen:
                    continue
                seen.add(smi)

                try:
                    scored = score_molecule(smi, models)
                    scored["parent_seed"] = parent_smiles
                    scored["round"] = round_idx
                    candidates.append(scored)
                except Exception:
                    continue

        if not candidates:
            print("[WARN] No candidates generated in this round.")
            break

        cand_df = pd.DataFrame(candidates).drop_duplicates(subset=["smiles"]).copy()

        # quality filters
        cand_df = cand_df[
            (cand_df["predicted_pIC50"] >= 8.5) &
            (cand_df["QED"] >= 0.40) &
            (cand_df["uncertainty"] <= 0.08) &
            (cand_df["penalty"] <= 0.3)
        ].copy()

        if cand_df.empty:
            print("[WARN] No candidates survived filters.")
            break

        cand_df = cand_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        all_generated.append(cand_df)

        # beam search: keep only top candidates for next round
        current_pool = cand_df.head(beam_width).copy()

        print("[INFO] Top round candidates:")
        print(current_pool[["smiles", "predicted_pIC50", "QED", "uncertainty", "final_score"]].head(10).to_string(index=False))

    if not all_generated:
        print("[WARN] No optimized molecules were generated.")
        return

    out = pd.concat(all_generated, ignore_index=True)
    out = out.drop_duplicates(subset=["smiles"]).sort_values("final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"
    out.to_csv(out_path, index=False)

    print(f"\n[OK] Saved iterative optimization results: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()