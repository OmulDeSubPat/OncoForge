from __future__ import annotations

from typing import Optional
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


_PAINS_CATALOG = build_pains_catalog()

_ALERT_SMARTS = {
    "nitroso": "[N]=O",
    "hydrazine": "[NX3][NX3]",
    "isocyanate": "N=C=O",
    "epoxide": "C1OC1",
    "acyl_halide": "[CX3](=O)[Cl,Br,I]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "aniline": "c[NH2]",
    "thiourea": "[NX3][CX3](=[SX1])[NX3]",
    "quinone": "O=C1C=CC(=O)C=C1",
}

_SEVERE_ALERTS = {"nitroso", "hydrazine", "isocyanate", "epoxide", "acyl_halide"}

_COVALENT_WARHEAD_SMARTS = {
    "acrylamide_like": "C=CC(=O)N",
    "propiolamide_like": "C#CC(=O)N",
    "vinyl_sulfone_like": "C=CS(=O)(=O)",
}

_ALERT_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in _ALERT_SMARTS.items()
}
_COVALENT_WARHEAD_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in _COVALENT_WARHEAD_SMARTS.items()
}


def pains_alert(smiles: str) -> tuple[bool, Optional[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True, "invalid_smiles"

    entry = _PAINS_CATALOG.GetFirstMatch(mol)
    if entry is None:
        return False, None

    return True, entry.GetDescription()


def structural_alerts(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["invalid_smiles"]

    matches = []
    for name, pattern in _ALERT_PATTERNS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matches.append(name)
    return matches


def severe_structural_alerts(smiles: str) -> list[str]:
    return [name for name in structural_alerts(smiles) if name in _SEVERE_ALERTS]


def covalent_warhead_alerts(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    matches = []
    for name, pattern in _COVALENT_WARHEAD_PATTERNS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matches.append(name)
    return matches
