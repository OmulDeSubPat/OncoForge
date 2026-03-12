from __future__ import annotations

from typing import List, Set
from rdkit import Chem


# substituții simple, sigure, pe care le încercăm în SMILES
SIMPLE_REPLACEMENTS = [
    ("Br", "Cl"),
    ("Cl", "Br"),
    ("F", "Cl"),
    ("Cl", "F"),
    ("C", "N"),
    ("N", "C"),
    ("OC", "NC"),
    ("NC", "OC"),
]


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def generate_string_mutations(smiles: str, max_variants: int = 50) -> List[str]:
    """
    Generează analogi simpli prin înlocuiri textuale controlate.
    Nu e chimie perfectă, dar este un generator bun de MVP.
    """
    variants: Set[str] = set()

    for old, new in SIMPLE_REPLACEMENTS:
        if old in smiles:
            mutated = smiles.replace(old, new, 1)
            can = canonicalize(mutated)
            if can is not None:
                variants.add(can)

    # mici extensii simple
    extensions = ["C", "F", "Cl", "Br", "N"]
    for ext in extensions:
        mutated = smiles + ext
        can = canonicalize(mutated)
        if can is not None:
            variants.add(can)

    variants.discard(smiles)

    out = list(variants)
    return out[:max_variants]