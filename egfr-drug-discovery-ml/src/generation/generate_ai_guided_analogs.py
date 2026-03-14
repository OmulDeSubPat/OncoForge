from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.generation.rgroup_generator import generate_rgroup_variants
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.chem import canonicalize_smiles


def main():
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"

    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    ranked_df = load_csv_artifact(
        ranked_path,
        required_columns=["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "audit_pass", "veto", "final_score"],
        producer="python -m src.models.rank_dataset",
    )
    scorer = build_default_scorer()

    seeds_df = ranked_df[
        (ranked_df["predicted_pIC50"] >= 8.5)
        & (ranked_df["QED"] >= 0.40)
        & (ranked_df["reward_hacking_risk"] <= 0.30)
        & (ranked_df["agent_disagreement_score"] <= 0.45)
        & (ranked_df["audit_pass"] == True)
        & (ranked_df["veto"] == False)
    ].copy()

    seeds_df = seeds_df.sort_values("final_score", ascending=False).head(15)
    seed_smiles = seeds_df["smiles"].tolist()

    generated_pairs = []
    seen = set()

    print(f"[INFO] Selected {len(seed_smiles)} high-quality seed molecules")

    for seed in tqdm(seed_smiles, desc="AI-guided analog generation"):
        variants = generate_rgroup_variants(seed, max_variants=120)

        for smi in variants:
            canonical_smiles = canonicalize_smiles(smi)
            if not canonical_smiles or canonical_smiles in seen:
                continue
            seen.add(canonical_smiles)
            generated_pairs.append({"smiles": canonical_smiles, "parent_seed": seed})

    if not generated_pairs:
        print("[WARN] No valid AI-guided analogs generated.")
        return

    generated_df = pd.DataFrame(generated_pairs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    out = score_smiles_list(generated_df["smiles"].tolist(), scorer=scorer)
    out = out.merge(generated_df, on="smiles", how="left")

    out_path = PROJECT_ROOT / "reports" / "ai_guided_analogs.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved AI-guided analogs: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
