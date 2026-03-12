from __future__ import annotations

import json
import joblib
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule
from src.generation.rgroup_generator import generate_rgroup_variants


def select_seed_pool(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    df = df.copy()
    df = df[
        (df["predicted_pIC50"] >= 8.5) &
        (df["QED"] >= 0.40) &
        (df["uncertainty"] <= 0.08) &
        (df["penalty"] <= 0.3)
    ].copy()

    if "SA_score" in df.columns:
        df = df[df["SA_score"] <= 4.5].copy()

    if "has_PAINS" in df.columns:
        df = df[df["has_PAINS"] == False].copy()

    return df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def save_checkpoint(out_dir: Path, round_idx: int, current_pool: pd.DataFrame, all_df: pd.DataFrame, seen: set[str]):
    out_dir.mkdir(parents=True, exist_ok=True)

    current_pool.to_csv(out_dir / f"checkpoint_round_{round_idx}_pool.csv", index=False)
    all_df.to_csv(out_dir / "all_generated_so_far.csv", index=False)

    with open(out_dir / "seen_smiles.json", "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen)), f, ensure_ascii=False, indent=2)


def load_seen(out_dir: Path) -> set[str]:
    p = out_dir / "seen_smiles.json"
    if not p.exists():
        return set()

    with open(p, "r", encoding="utf-8") as f:
        return set(json.load(f))


def main():
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"
    out_dir = PROJECT_ROOT / "reports" / "long_run_generation"

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

    # parametri long-run
    n_rounds = 12
    beam_width = 20
    variants_per_seed = 120
    top_global_keep = 1000

    current_pool = select_seed_pool(ranked_df, top_k=beam_width)
    seen = set(current_pool["smiles"].tolist())
    seen |= load_seen(out_dir)

    all_generated_list = []

    print(f"[INFO] Starting long-run AI generation")
    print(f"[INFO] Initial seeds: {len(current_pool)}")
    print(f"[INFO] Rounds: {n_rounds}, Beam width: {beam_width}, Variants/seed: {variants_per_seed}")

    for round_idx in range(1, n_rounds + 1):
        print(f"\n[INFO] Round {round_idx}")

        round_candidates = []

        for _, row in tqdm(current_pool.iterrows(), total=len(current_pool), desc=f"Round {round_idx}"):
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
                    round_candidates.append(scored)
                except Exception:
                    continue

        if not round_candidates:
            print("[WARN] No candidates generated this round.")
            break

        round_df = pd.DataFrame(round_candidates).drop_duplicates(subset=["smiles"]).copy()

        # filtre de calitate
        round_df = round_df[
            (round_df["predicted_pIC50"] >= 8.3) &
            (round_df["QED"] >= 0.35) &
            (round_df["uncertainty"] <= 0.10) &
            (round_df["penalty"] <= 0.3)
        ].copy()

        if "SA_score" in round_df.columns:
            round_df = round_df[round_df["SA_score"] <= 5.0].copy()

        if "has_PAINS" in round_df.columns:
            round_df = round_df[round_df["has_PAINS"] == False].copy()

        if round_df.empty:
            print("[WARN] No candidates survived quality filters.")
            break

        round_df = round_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        all_generated_list.append(round_df)

        all_df = pd.concat(all_generated_list, ignore_index=True)
        all_df = (
            all_df.drop_duplicates(subset=["smiles"])
            .sort_values("final_score", ascending=False)
            .head(top_global_keep)
            .reset_index(drop=True)
        )

        current_pool = all_df.head(beam_width).copy()

        save_checkpoint(out_dir, round_idx, current_pool, all_df, seen)

        print("[INFO] Top candidates after round:")
        print(
            current_pool[
                ["smiles", "predicted_pIC50", "QED", "uncertainty", "SA_score", "final_score"]
            ].head(10).to_string(index=False)
        )

    if not all_generated_list:
        print("[WARN] No long-run candidates produced.")
        return

    final_df = pd.concat(all_generated_list, ignore_index=True)
    final_df = (
        final_df.drop_duplicates(subset=["smiles"])
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
    )

    out_path = out_dir / "long_run_final_candidates.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\n[OK] Saved long-run final candidates: {out_path}")
    print(final_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()