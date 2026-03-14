from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan/ECFP bit-vector fingerprint as numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if radius == 2 and n_bits == 2048:
        fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
