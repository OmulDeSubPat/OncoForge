from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from src.config import PROJECT_ROOT


def main():
    in_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"
    if not in_path.exists():
        in_path = PROJECT_ROOT / "reports" / "final_diverse_candidates.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            "Missing candidate file. Run iterative optimizer or candidate selection first."
        )

    df = pd.read_csv(in_path).head(12)

    mols = []
    legends = []

    for _, row in df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mols.append(mol)
        legends.append(
            f"Score={row['final_score']:.2f}\npIC50={row['predicted_pIC50']:.2f}\nQED={row['QED']:.2f}"
        )

    if not mols:
        raise ValueError("No valid molecules to draw.")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(350, 300),
        legends=legends,
        useSVG=False,
    )

    out_path = PROJECT_ROOT / "reports" / "top_molecules_grid.png"
    img.save(str(out_path))

    print(f"[OK] Saved molecule grid: {out_path}")


if __name__ == "__main__":
    main()