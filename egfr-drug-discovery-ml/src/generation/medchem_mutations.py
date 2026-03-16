from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem, rdBase

from src.generation.generator_policy import apply_generator_policy
from src.generation.matched_molecular_transformations import generate_mmp_outcomes
from src.generation.reaction_transformations import generate_reaction_outcomes
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


@dataclass(frozen=True)
class MutationOutcome:
    action_name: str
    smiles: str
    category: str
    rule_source: str = "medchem_edit"
    reaction_family: str = "medchem_edit"
    synthetic_route: str | None = None
    synthetic_feasibility_score: float = 0.50
    medchem_realism_score: float = 0.50
    transformation_confidence: float = 0.50
    preserves_scaffold: bool = True
    parent_similarity: float = 0.0
    property_support_score: float = 0.0
    category_priority_score: float = 0.0
    generator_priority_score: float = 0.0
    hard_constraint_pass: bool = True
    hard_constraint_notes: str | None = None
    introduced_warhead: bool = False
    warhead_retained: bool = True
    alert_count: int = 0
    severe_alert_count: int = 0


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
    return [outcome.smiles for outcome in generate_medchem_outcomes(smiles, max_variants=max_variants)]


def generate_medchem_outcomes(smiles: str, max_variants: int = 100) -> list[MutationOutcome]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []

    outcomes: dict[str, MutationOutcome] = {}

    for atom_idx in terminal_halogen_sites(mol):
        atom = mol.GetAtomWithIdx(atom_idx)
        for atomic_num in HALOGEN_SWAPS.get(atom.GetAtomicNum(), []):
            candidate = _replace_atom(mol, atom_idx, atomic_num)
            if candidate is not None and candidate != smiles:
                outcomes[candidate] = MutationOutcome(
                    action_name=f"halogen_swap_{atom.GetAtomicNum()}_to_{atomic_num}",
                    smiles=candidate,
                    category="atom_swap",
                    rule_source="atom_edit",
                    reaction_family="halogen_scan",
                    synthetic_route="late_stage_halogen_exchange",
                    synthetic_feasibility_score=0.83,
                    medchem_realism_score=0.80,
                    transformation_confidence=0.76,
                )

    for atom_idx in aromatic_attachment_sites(mol):
        for group_name, group in ATTACHMENT_GROUPS.items():
            candidate = _attach_group(mol, atom_idx, group)
            if candidate is not None and candidate != smiles:
                outcomes[candidate] = MutationOutcome(
                    action_name=f"append_{group_name}",
                    smiles=candidate,
                    category="append_group",
                    rule_source="append_group",
                    reaction_family="late_stage_diversification",
                    synthetic_route="append_group_scan",
                    synthetic_feasibility_score=0.68 if group_name in {"methoxy", "amino"} else 0.74,
                    medchem_realism_score=0.70 if group_name in {"methoxy", "amino"} else 0.63,
                    transformation_confidence=0.64,
                )

    for atom_idx in hetero_methylation_sites(mol):
        candidate = _methylate_atom(mol, atom_idx)
        if candidate is not None and candidate != smiles:
                outcomes[candidate] = MutationOutcome(
                    action_name="hetero_methylation",
                    smiles=candidate,
                    category="hetero_edit",
                    rule_source="hetero_edit",
                    reaction_family="n_or_o_alkylation",
                    synthetic_route="heteroatom_methylation",
                    synthetic_feasibility_score=0.86,
                    medchem_realism_score=0.82,
                    transformation_confidence=0.84,
                )

    for outcome in generate_mmp_outcomes(smiles, max_variants=max_variants):
        if outcome.smiles != smiles:
            outcomes[outcome.smiles] = MutationOutcome(
                action_name=outcome.action_name,
                smiles=outcome.smiles,
                category=outcome.category,
                rule_source=outcome.rule_source,
                reaction_family=outcome.reaction_family,
                synthetic_route=outcome.synthetic_route,
                synthetic_feasibility_score=outcome.synthetic_feasibility_score,
                medchem_realism_score=outcome.medchem_realism_score,
                transformation_confidence=outcome.transformation_confidence,
                preserves_scaffold=outcome.preserves_scaffold,
            )

    for outcome in generate_reaction_outcomes(smiles, max_variants=max_variants):
        if outcome.smiles != smiles:
            outcomes[outcome.smiles] = MutationOutcome(
                action_name=outcome.action_name,
                smiles=outcome.smiles,
                category=outcome.category,
                rule_source=outcome.rule_source,
                reaction_family=outcome.reaction_family,
                synthetic_route=outcome.synthetic_route,
                synthetic_feasibility_score=outcome.synthetic_feasibility_score,
                medchem_realism_score=outcome.medchem_realism_score,
                transformation_confidence=outcome.transformation_confidence,
                preserves_scaffold=outcome.preserves_scaffold,
            )

    filtered = apply_generator_policy(smiles, outcomes.values(), max_variants=max_variants)
    return filtered[:max_variants]
