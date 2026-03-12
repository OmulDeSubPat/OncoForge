from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str) -> str:
    """
    Return canonical Murcko scaffold SMILES for a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles_canonical",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by Murcko scaffolds: no scaffold overlap between train and test.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col='{smiles_col}' in df.columns")

    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(murcko_scaffold)

    scaffolds = df["scaffold"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)

    # Count rows per scaffold
    scaffold_to_rows = df.groupby("scaffold").size().to_dict()

    target_test = int(len(df) * test_size)
    test_scaffolds = set()
    test_count = 0

    for scaf in scaffolds:
        if test_count >= target_test:
            break
        test_scaffolds.add(scaf)
        test_count += scaffold_to_rows.get(scaf, 0)

    test_df = df[df["scaffold"].isin(test_scaffolds)].drop(columns=["scaffold"])
    train_df = df[~df["scaffold"].isin(test_scaffolds)].drop(columns=["scaffold"])

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
