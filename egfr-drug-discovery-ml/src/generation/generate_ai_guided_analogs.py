from __future__ import annotations

import joblib
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule
from src.generation.rgroup_generator import generate_rgroup_variants


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
            f"Missing ensemble model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )

    ranked_df = pd.read_csv(ranked_path)
    models = joblib.load(model_path)

    # Seed selection = AI-guided starting points
    # luăm molecule bune și relativ drug-like
    seeds_df = ranked_df[
        (ranked_df["predicted_pIC50"] >= 8.8) &
        (ranked_df["QED"] >= 0.45) &
        (ranked_df["uncertainty"] <= 0.05)
    ].copy()

    seeds_df = seeds_df.sort_values("final_score", ascending=False).head(15)

    seed_smiles = seeds_df["smiles"].tolist()

    generated_rows = []
    seen = set()

    print(f"[INFO] Selected {len(seed_smiles)} high-quality seed molecules")

    for seed in tqdm(seed_smiles, desc="AI-guided analog generation"):
        variants = generate_rgroup_variants(seed, max_variants=100)

        for smi in variants:
            if smi in seen:
                continue
            seen.add(smi)

            try:
                scored = score_molecule(smi, models)
                scored["parent_seed"] = seed
                generated_rows.append(scored)
            except Exception:
                continue

    if not generated_rows:
        print("[WARN] No valid AI-guided analogs generated.")
        return

    out = pd.DataFrame(generated_rows).drop_duplicates(subset=["smiles"]).copy()
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "ai_guided_analogs.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved AI-guided analogs: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()