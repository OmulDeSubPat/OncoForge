from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.generation.rgroup_generator import generate_rgroup_variants
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.chem import canonicalize_smiles


def select_seed_pool(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    filtered = df[
        (df["predicted_pIC50"] >= 8.5)
        & (df["QED"] >= 0.40)
        & (df["reward_hacking_risk"] <= 0.30)
        & (df["agent_disagreement_score"] <= 0.45)
        & (df["applicability_score"] >= 0.30)
        & (df["audit_pass"] == True)
        & (df["veto"] == False)
    ].copy()
    return filtered.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def main():
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    ranked_df = load_csv_artifact(
        ranked_path,
        required_columns=["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "applicability_score", "audit_pass", "veto", "final_score"],
        producer="python -m src.models.rank_dataset",
    )
    scorer = build_default_scorer()

    current_pool = select_seed_pool(ranked_df, top_k=10)
    all_generated = []
    seen = set(current_pool["smiles"].tolist())

    n_rounds = 4
    beam_width = 12
    variants_per_seed = 80

    print(f"[INFO] Starting iterative optimization with {len(current_pool)} seed molecules")

    for round_idx in range(1, n_rounds + 1):
        print(f"\n[INFO] Round {round_idx}")

        candidate_pairs = []

        for _, row in tqdm(current_pool.iterrows(), total=len(current_pool), desc=f"Round {round_idx} generation"):
            parent_smiles = row["smiles"]
            variants = generate_rgroup_variants(parent_smiles, max_variants=variants_per_seed)

            for smi in variants:
                canonical_smiles = canonicalize_smiles(smi)
                if not canonical_smiles or canonical_smiles in seen:
                    continue
                seen.add(canonical_smiles)
                candidate_pairs.append(
                    {
                        "smiles": canonical_smiles,
                        "parent_seed": parent_smiles,
                        "round": round_idx,
                    }
                )

        if not candidate_pairs:
            print("[WARN] No candidates generated in this round.")
            break

        candidate_df = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        cand_df = score_smiles_list(candidate_df["smiles"].tolist(), scorer=scorer)
        cand_df = cand_df.merge(candidate_df, on="smiles", how="left")
        cand_df = cand_df[
            (cand_df["predicted_pIC50"] >= 8.3)
            & (cand_df["QED"] >= 0.35)
            & (cand_df["reward_hacking_risk"] <= 0.45)
            & (cand_df["agent_disagreement_score"] <= 0.55)
            & (cand_df["audit_status"] != "fail")
            & (cand_df["veto"] == False)
        ].copy()

        if cand_df.empty:
            print("[WARN] No candidates survived filters.")
            break

        all_generated.append(cand_df)
        current_pool = cand_df.head(beam_width).copy()

        print("[INFO] Top round candidates:")
        print(
            current_pool[
                ["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "final_score"]
            ].head(10).to_string(index=False)
        )

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
