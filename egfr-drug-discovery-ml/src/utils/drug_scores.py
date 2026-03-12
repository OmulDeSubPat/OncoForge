from __future__ import annotations

from typing import Optional
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, rdMolDescriptors


def qed_score(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(QED.qed(mol))


def molecular_weight(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def logp(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Crippen.MolLogP(mol))


def tpsa(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(rdMolDescriptors.CalcTPSA(mol))


def hbd(smiles: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(rdMolDescriptors.CalcNumHBD(mol))


def hba(smiles: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(rdMolDescriptors.CalcNumHBA(mol))