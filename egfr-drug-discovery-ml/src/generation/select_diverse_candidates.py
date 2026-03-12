from __future__ import annotations

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.config import PROJECT_ROOT


def fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2) -> float:
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def main():
    in_path = PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing file: {in_path}\n"
            "Run: python -m src.generation.generate_and_rank_analogs"
        )

    df = pd.read_csv(in_path)

    # basic quality filters
    df = df[
        (df["predicted_pIC50"] >= 8.5) &
        (df["QED"] >= 0.45) &
        (df["uncertainty"] <= 0.05) &
        (df["penalty"] <= 0.3)
    ].copy()

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected = []
    selected_fps = []

    similarity_threshold = 0.75
    max_candidates = 20

    for _, row in df.iterrows():
        smi = row["smiles"]
        fp = fingerprint(smi)
        if fp is None:
            continue

        too_similar = False
        for prev_fp in selected_fps:
            if tanimoto(fp, prev_fp) >= similarity_threshold:
                too_similar = True
                break

        if not too_similar:
            selected.append(row.to_dict())
            selected_fps.append(fp)

        if len(selected) >= max_candidates:
            break

    out = pd.DataFrame(selected)

    out_path = PROJECT_ROOT / "reports" / "final_diverse_candidates.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved final diverse candidates: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()