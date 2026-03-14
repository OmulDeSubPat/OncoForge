from __future__ import annotations

from typing import Iterable

from rdkit import Chem, rdBase

from src.generation.matched_molecular_transformations import generate_mmp_variants
from src.utils.similarity import mol_from_smiles


HALOGEN_SWAPS = {
    9: [17, 35],
    17: [9, 35],
    35: [9, 17],
}

ATTACHMENT_GROUPS = {
    "F": [9],
    "Cl": [17],
    "methyl": [6],
    "amino": [7],
    "methoxy": [8, 6],
}


def canonicalize_mol(mol) -> str | None:
    if mol is None:
        return None
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Exception:
        return None
    with rdBase.BlockLogs():
        return Chem.MolToSmiles(mol, canonical=True)


def is_reasonable_molecule(mol, max_atoms: int = 80) -> bool:
    if mol is None:
        return False
    if mol.GetNumAtoms() > max_atoms:
        return False
    if mol.GetNumHeavyAtoms() < 6:
        return False
    return True


def aromatic_attachment_sites(mol) -> list[int]:
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if not atom.GetIsAromatic():
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if atom.GetDegree() > 3:
            continue
        sites.append(atom.GetIdx())
    return sites


def terminal_halogen_sites(mol) -> list[int]:
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in HALOGEN_SWAPS:
            continue
        if atom.GetDegree() != 1:
            continue
        sites.append(atom.GetIdx())
    return sites


def hetero_methylation_sites(mol) -> list[int]:
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {7, 8}:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if atom.GetDegree() > 2:
            continue
        sites.append(atom.GetIdx())
    return sites


def _attach_group(mol, atom_idx: int, atomic_nums: Iterable[int]) -> str | None:
    rw = Chem.RWMol(mol)
    previous_idx = atom_idx

    for atomic_num in atomic_nums:
        new_idx = rw.AddAtom(Chem.Atom(int(atomic_num)))
        rw.AddBond(previous_idx, new_idx, Chem.BondType.SINGLE)
        previous_idx = new_idx

    candidate = rw.GetMol()
    if not is_reasonable_molecule(candidate):
        return None
    return canonicalize_mol(candidate)


def _replace_atom(mol, atom_idx: int, atomic_num: int) -> str | None:
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(atom_idx).SetAtomicNum(int(atomic_num))
    candidate = rw.GetMol()
    if not is_reasonable_molecule(candidate):
        return None
    return canonicalize_mol(candidate)


def _methylate_atom(mol, atom_idx: int) -> str | None:
    rw = Chem.RWMol(mol)
    carbon_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(atom_idx, carbon_idx, Chem.BondType.SINGLE)
    candidate = rw.GetMol()
    if not is_reasonable_molecule(candidate):
        return None
    return canonicalize_mol(candidate)


def generate_medchem_variants(smiles: str, max_variants: int = 100) -> list[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []

    variants: set[str] = set()

    for atom_idx in terminal_halogen_sites(mol):
        atom = mol.GetAtomWithIdx(atom_idx)
        for atomic_num in HALOGEN_SWAPS.get(atom.GetAtomicNum(), []):
            candidate = _replace_atom(mol, atom_idx, atomic_num)
            if candidate is not None and candidate != smiles:
                variants.add(candidate)

    for atom_idx in aromatic_attachment_sites(mol):
        for group in ATTACHMENT_GROUPS.values():
            candidate = _attach_group(mol, atom_idx, group)
            if candidate is not None and candidate != smiles:
                variants.add(candidate)

    for atom_idx in hetero_methylation_sites(mol):
        candidate = _methylate_atom(mol, atom_idx)
        if candidate is not None and candidate != smiles:
            variants.add(candidate)

    for candidate in generate_mmp_variants(smiles, max_variants=max_variants):
        if candidate != smiles:
            variants.add(candidate)

    return sorted(variants)[:max_variants]
