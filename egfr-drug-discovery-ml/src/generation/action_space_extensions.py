from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from src.utils.similarity import mol_from_smiles, murcko_scaffold_smiles


@dataclass(frozen=True)
class ActionSpaceOutcome:
    action_name: str
    smiles: str
    category: str
    rule_source: str
    reaction_family: str
    synthetic_route: str | None
    synthetic_feasibility_score: float
    medchem_realism_score: float
    transformation_confidence: float
    preserves_scaffold: bool = True


@dataclass(frozen=True)
class FragmentTemplate:
    name: str
    fragment_smiles: str
    attach_idx: int
    category: str
    synthetic_route: str
    synthetic_feasibility_score: float
    medchem_realism_score: float
    transformation_confidence: float


@dataclass(frozen=True)
class LinkerRule:
    name: str
    reaction_smarts: str
    synthetic_route: str
    synthetic_feasibility_score: float
    medchem_realism_score: float
    transformation_confidence: float


FRAGMENT_GROWTH_LIBRARY = [
    FragmentTemplate("grow_nitrile_tail", "C#N", 0, "fragment_growing", "late_stage_nitrile_append", 0.74, 0.72, 0.70),
    FragmentTemplate("grow_difluoromethyl_tail", "C(F)F", 0, "fragment_growing", "halogenated_tail_scan", 0.71, 0.69, 0.66),
    FragmentTemplate("grow_methoxy_tail", "OC", 0, "fragment_growing", "o_alkylation", 0.84, 0.78, 0.75),
    FragmentTemplate("grow_methylamino_tail", "NC", 0, "fragment_growing", "late_stage_amination", 0.79, 0.81, 0.76),
    FragmentTemplate("grow_cyclopropyl_tail", "C1CC1", 0, "fragment_growing", "cyclopropyl_installation", 0.73, 0.74, 0.70),
    FragmentTemplate("grow_morpholine_tail", "N1CCOCC1", 0, "fragment_growing", "solubilizing_tail_installation", 0.67, 0.72, 0.64),
]

SCAFFOLD_DECORATION_LIBRARY = [
    FragmentTemplate("decorate_fluoro", "F", 0, "scaffold_decoration", "late_stage_fluorination", 0.88, 0.82, 0.80),
    FragmentTemplate("decorate_chloro", "Cl", 0, "scaffold_decoration", "late_stage_chlorination", 0.84, 0.79, 0.77),
    FragmentTemplate("decorate_cyano", "C#N", 0, "scaffold_decoration", "snar_cyano_diversification", 0.77, 0.76, 0.74),
    FragmentTemplate("decorate_trifluoromethyl", "C(F)(F)F", 0, "scaffold_decoration", "trifluoromethyl_scan", 0.72, 0.74, 0.70),
    FragmentTemplate("decorate_methoxy", "OC", 0, "scaffold_decoration", "late_stage_methoxylation", 0.85, 0.80, 0.79),
    FragmentTemplate("decorate_methylamino", "NC", 0, "scaffold_decoration", "snar_amination", 0.83, 0.82, 0.80),
]

LINKER_REPLACEMENT_RULES = [
    LinkerRule("linker_ether_to_amine", "[*:1]CO[*:2]>>[*:1]CN[*:2]", "ether_to_amine_linker_swap", 0.72, 0.73, 0.69),
    LinkerRule("linker_amine_to_ether", "[*:1]CN[*:2]>>[*:1]CO[*:2]", "amine_to_ether_linker_swap", 0.74, 0.71, 0.68),
    LinkerRule("linker_methylene_to_ether", "[*:1]CC[*:2]>>[*:1]CO[*:2]", "methylene_to_ether_linker_swap", 0.70, 0.68, 0.64),
    LinkerRule("linker_ether_to_thioether", "[*:1]CO[*:2]>>[*:1]CS[*:2]", "ether_to_thioether_swap", 0.68, 0.66, 0.62),
    LinkerRule("linker_amide_to_urea", "[*:1]NC(=O)[*:2]>>[*:1]NC(=O)N[*:2]", "amide_to_urea_extension", 0.75, 0.78, 0.72),
]


def _sanitize_candidate(mol) -> str | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _same_core_scaffold(parent_smiles: str, candidate_smiles: str) -> bool:
    parent_scaffold = murcko_scaffold_smiles(parent_smiles)
    candidate_scaffold = murcko_scaffold_smiles(candidate_smiles)
    return bool(parent_scaffold) and parent_scaffold == candidate_scaffold


def _reasonable_candidate(mol) -> bool:
    return mol is not None and 6 <= mol.GetNumHeavyAtoms() <= 80


def _aromatic_h_sites(mol) -> list[int]:
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic():
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if atom.GetDegree() > 3:
            continue
        sites.append(atom.GetIdx())
    return sites


def _hetero_growth_sites(mol) -> list[int]:
    sites: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {7, 8}:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetDegree() > 2:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        sites.append(atom.GetIdx())
    return sites


def _attach_fragment(parent, atom_idx: int, fragment_smiles: str, attach_idx: int = 0) -> str | None:
    fragment = mol_from_smiles(fragment_smiles)
    if fragment is None:
        return None
    combo = Chem.CombineMols(parent, fragment)
    rw = Chem.RWMol(combo)
    parent_atoms = parent.GetNumAtoms()
    rw.AddBond(atom_idx, parent_atoms + int(attach_idx), Chem.BondType.SINGLE)
    candidate = rw.GetMol()
    if not _reasonable_candidate(candidate):
        return None
    return _sanitize_candidate(candidate)


def _compile_linker_rule(rule: LinkerRule):
    with rdBase.BlockLogs():
        return AllChem.ReactionFromSmarts(rule.reaction_smarts)


def generate_fragment_growing_outcomes(smiles: str, max_variants: int = 36) -> list[ActionSpaceOutcome]:
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []
    outcomes: dict[str, ActionSpaceOutcome] = {}
    candidate_sites = list(dict.fromkeys(_aromatic_h_sites(parent)[:4] + _hetero_growth_sites(parent)[:3]))
    for atom_idx in candidate_sites:
        for template in FRAGMENT_GROWTH_LIBRARY:
            candidate = _attach_fragment(parent, atom_idx, template.fragment_smiles, attach_idx=template.attach_idx)
            if not candidate or candidate == smiles:
                continue
            if not _same_core_scaffold(smiles, candidate):
                continue
            outcomes[candidate] = ActionSpaceOutcome(
                action_name=template.name,
                smiles=candidate,
                category=template.category,
                rule_source="fragment_growing",
                reaction_family="fragment_growing",
                synthetic_route=template.synthetic_route,
                synthetic_feasibility_score=template.synthetic_feasibility_score,
                medchem_realism_score=template.medchem_realism_score,
                transformation_confidence=template.transformation_confidence,
                preserves_scaffold=True,
            )
            if len(outcomes) >= max_variants:
                return sorted(outcomes.values(), key=lambda item: item.smiles)
    return sorted(outcomes.values(), key=lambda item: item.smiles)


def generate_scaffold_decoration_outcomes(smiles: str, max_variants: int = 36) -> list[ActionSpaceOutcome]:
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []
    outcomes: dict[str, ActionSpaceOutcome] = {}
    for atom_idx in _aromatic_h_sites(parent)[:6]:
        for template in SCAFFOLD_DECORATION_LIBRARY:
            candidate = _attach_fragment(parent, atom_idx, template.fragment_smiles, attach_idx=template.attach_idx)
            if not candidate or candidate == smiles:
                continue
            if not _same_core_scaffold(smiles, candidate):
                continue
            outcomes[candidate] = ActionSpaceOutcome(
                action_name=template.name,
                smiles=candidate,
                category=template.category,
                rule_source="scaffold_decoration",
                reaction_family="scaffold_decoration",
                synthetic_route=template.synthetic_route,
                synthetic_feasibility_score=template.synthetic_feasibility_score,
                medchem_realism_score=template.medchem_realism_score,
                transformation_confidence=template.transformation_confidence,
                preserves_scaffold=True,
            )
            if len(outcomes) >= max_variants:
                return sorted(outcomes.values(), key=lambda item: item.smiles)
    return sorted(outcomes.values(), key=lambda item: item.smiles)


def generate_linker_replacement_outcomes(smiles: str, max_variants: int = 28) -> list[ActionSpaceOutcome]:
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []
    outcomes: dict[str, ActionSpaceOutcome] = {}
    for rule in LINKER_REPLACEMENT_RULES:
        reaction = _compile_linker_rule(rule)
        with rdBase.BlockLogs():
            products = reaction.RunReactants((parent,))
        for product_tuple in products:
            if not product_tuple:
                continue
            candidate = _sanitize_candidate(product_tuple[0])
            if not candidate or candidate == smiles:
                continue
            if not _same_core_scaffold(smiles, candidate):
                continue
            outcomes[candidate] = ActionSpaceOutcome(
                action_name=rule.name,
                smiles=candidate,
                category="linker_replacement",
                rule_source="linker_replacement",
                reaction_family="linker_replacement",
                synthetic_route=rule.synthetic_route,
                synthetic_feasibility_score=rule.synthetic_feasibility_score,
                medchem_realism_score=rule.medchem_realism_score,
                transformation_confidence=rule.transformation_confidence,
                preserves_scaffold=True,
            )
            if len(outcomes) >= max_variants:
                return sorted(outcomes.values(), key=lambda item: item.smiles)
    return sorted(outcomes.values(), key=lambda item: item.smiles)
