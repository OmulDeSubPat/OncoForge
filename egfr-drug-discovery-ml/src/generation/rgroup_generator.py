from __future__ import annotations

from typing import List, Set
from rdkit import Chem


# Transformări mai "medchem-aware"
SUBSTITUTION_RULES = [
    ("Br", "Cl"),
    ("Br", "F"),
    ("Cl", "F"),
    ("F", "Cl"),
    ("F", "Br"),
    ("Cl", "Br"),

    ("C", "OC"),      # methyl -> methoxy-like extension in simple cases
    ("OC", "C"),
    ("N", "O"),
    ("O", "N"),
    ("CN", "CO"),
    ("CO", "CN"),
]


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_reasonable_size(smiles: str, max_atoms: int = 80) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= max_atoms


def generate_rgroup_variants(smiles: str, max_variants: int = 100) -> List[str]:
    """
    Generator simplu, dar mai inteligent decât mutațiile brute:
    aplică substituții mici, locale, pe fragmente periferice.
    """
    variants: Set[str] = set()

    for old, new in SUBSTITUTION_RULES:
        start = 0
        while True:
            idx = smiles.find(old, start)
            if idx == -1:
                break

            mutated = smiles[:idx] + new + smiles[idx + len(old):]
            can = canonicalize(mutated)

            if can is not None and can != smiles and is_reasonable_size(can):
                variants.add(can)

            start = idx + 1

    return list(variants)[:max_variants]