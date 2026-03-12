from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import os


def build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


_PAINS_CATALOG = build_pains_catalog()


def pains_alert(smiles: str) -> tuple[bool, Optional[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True, "invalid_smiles"

    entry = _PAINS_CATALOG.GetFirstMatch(mol)
    if entry is None:
        return False, None

    return True, entry.GetDescription()