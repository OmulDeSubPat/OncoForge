from __future__ import annotations

import joblib
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule
from src.generation.analog_generator import generate_string_mutations


def main():
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ensemble model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )

    models = joblib.load(model_path)
    ranked_df = pd.read_csv(ranked_path)

    # luăm primele 20 molecule ca seeds
    top_n = 20
    seed_smiles = ranked_df["smiles"].head(top_n).tolist()

    generated_rows = []
    seen = set()

    print(f"[INFO] Using top {top_n} seed molecules")
    for seed in tqdm(seed_smiles, desc="Generating analogs"):
        analogs = generate_string_mutations(seed, max_variants=50)

        for analog in analogs:
            if analog in seen:
                continue
            seen.add(analog)

            try:
                scored = score_molecule(analog, models)
                scored["parent_seed"] = seed
                generated_rows.append(scored)
            except Exception:
                # ignoră analogii care pică pe parsing/scoring
                continue

    if not generated_rows:
        print("[WARN] No valid analogs generated.")
        return

    out = pd.DataFrame(generated_rows)

    # scoatem duplicate după smiles
    out = out.drop_duplicates(subset=["smiles"]).copy()

    # sortare după scor
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "generated_analogs_ranked.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved generated analogs: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()