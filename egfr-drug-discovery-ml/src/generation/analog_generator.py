from __future__ import annotations

from typing import List

from src.generation.medchem_mutations import generate_medchem_variants
from src.utils.similarity import mol_from_smiles


def generate_string_mutations(smiles: str, max_variants: int = 50) -> List[str]:
    """
    Generate chemically controlled analogs using RDKit graph edits.
    The old function name is preserved for backward compatibility.
    """
    if mol_from_smiles(smiles) is None:
        return []
    return generate_medchem_variants(smiles, max_variants=max_variants)
