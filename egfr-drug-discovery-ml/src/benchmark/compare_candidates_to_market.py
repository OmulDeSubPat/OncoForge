from __future__ import annotations

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.config import PROJECT_ROOT


def fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def main():
    candidates_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"
    market_path = PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv"

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidates file: {candidates_path}\n"
            "Run: python -m src.generation.iterative_ai_optimizer"
        )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Missing market benchmark file: {market_path}\n"
            "Run: python -m src.benchmark.score_marketed_egfr"
        )

    cand = pd.read_csv(candidates_path).copy()
    market = pd.read_csv(market_path).copy()

    market_fps = []
    for _, row in market.iterrows():
        mfp = fp(row["smiles"])
        if mfp is not None:
            market_fps.append((row["name"], row["smiles"], mfp))

    rows = []
    for _, row in cand.iterrows():
        cfp = fp(row["smiles"])
        if cfp is None:
            continue

        best_name = None
        best_smiles = None
        best_sim = -1.0

        for m_name, m_smiles, mfp in market_fps:
            sim = float(DataStructs.TanimotoSimilarity(cfp, mfp))
            if sim > best_sim:
                best_sim = sim
                best_name = m_name
                best_smiles = m_smiles

        out_row = row.to_dict()
        out_row["closest_market_drug"] = best_name
        out_row["closest_market_smiles"] = best_smiles
        out_row["max_market_similarity"] = best_sim
        rows.append(out_row)

    out = pd.DataFrame(rows).sort_values(
        ["final_score", "max_market_similarity"],
        ascending=[False, True]
    ).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "candidates_vs_market.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved candidate vs market comparison: {out_path}")
    print(
        out[
            [
                "smiles",
                "predicted_pIC50",
                "QED",
                "uncertainty",
                "final_score",
                "closest_market_drug",
                "max_market_similarity",
                "round",
            ]
        ].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main()