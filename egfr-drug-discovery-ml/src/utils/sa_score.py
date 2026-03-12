from __future__ import annotations

from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def simple_sa_score(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    score = 1.0
    score += max(0, (mw - 350.0) / 150.0)
    score += 0.20 * ring_count
    score += 0.35 * chiral_centers
    score += 0.50 * spiro
    score += 0.50 * bridge
    score += 0.10 * rot

    return float(score)