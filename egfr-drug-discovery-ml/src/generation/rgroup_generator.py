from __future__ import annotations

from typing import List

from src.generation.medchem_mutations import generate_medchem_variants
from src.utils.similarity import mol_from_smiles


def generate_rgroup_variants(smiles: str, max_variants: int = 100) -> List[str]:
    """
    Generate scaffold-preserving peripheral variants around aromatic
    attachment sites and terminal groups using RDKit edits.
    """
    if mol_from_smiles(smiles) is None:
        return []
    return generate_medchem_variants(smiles, max_variants=max_variants)
