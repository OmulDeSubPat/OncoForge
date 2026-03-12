from typing import Optional
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover


remover = SaltRemover.SaltRemover()


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Parse SMILES, remove salts (keep largest fragment),
    return canonical SMILES.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = remover.StripMol(mol, dontRemoveEverything=True)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def ic50_nm_to_pic50(ic50_nm: float) -> Optional[float]:
    """
    Convert IC50 in nM to pIC50.
    pIC50 = 9 − log10(IC50 in nM)
    """

    if ic50_nm is None:
        return None

    if not np.isfinite(ic50_nm):
        return None

    if ic50_nm <= 0:
        return None

    return 9.0 - math.log10(ic50_nm)