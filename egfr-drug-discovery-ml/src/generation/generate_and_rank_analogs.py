from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.generation.analog_generator import generate_string_mutations
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
        required_columns=["smiles", "veto", "audit_pass", "reward_hacking_risk", "agent_disagreement_score", "applicability_score", "final_score"],
        producer="python -m src.models.rank_dataset",
    )
    scorer = build_default_scorer()

    seed_df = ranked_df[
        (ranked_df["veto"] == False)
        & (ranked_df["audit_pass"] == True)
        & (ranked_df["reward_hacking_risk"] <= 0.35)
        & (ranked_df["agent_disagreement_score"] <= 0.45)
        & (ranked_df["applicability_score"] >= 0.30)
    ].head(24)
    seed_smiles = seed_df["smiles"].tolist()

    generated_pairs = []
    seen = set()

    print(f"[INFO] Using {len(seed_smiles)} high-confidence seed molecules")
    for seed in tqdm(seed_smiles, desc="Generating analogs"):
        analogs = generate_string_mutations(seed, max_variants=60)

        for analog in analogs:
            canonical_smiles = canonicalize_smiles(analog)
            if not canonical_smiles or canonical_smiles in seen:
                continue
            seen.add(canonical_smiles)
            generated_pairs.append({"smiles": canonical_smiles, "parent_seed": seed})

    if not generated_pairs:
        print("[WARN] No valid analogs generated.")
        return

    generated_df = pd.DataFrame(generated_pairs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    out = score_smiles_list(generated_df["smiles"].tolist(), scorer=scorer)
    out = out.merge(generated_df, on="smiles", how="left")

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "generated_analogs_ranked.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved generated analogs: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
