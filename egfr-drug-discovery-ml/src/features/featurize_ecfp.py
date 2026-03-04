from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def ecfp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan/ECFP bit-vector fingerprint as numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
