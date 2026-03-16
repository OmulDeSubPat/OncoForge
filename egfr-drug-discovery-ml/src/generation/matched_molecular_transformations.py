from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from src.utils.similarity import mol_from_smiles, murcko_scaffold_smiles


@dataclass(frozen=True)
class TransformationRule:
    name: str
    reaction_smarts: str
    synthetic_route: str
    synthetic_feasibility_score: float
    medchem_realism_score: float
    transformation_confidence: float


@dataclass(frozen=True)
class TransformationOutcome:
    action_name: str
    smiles: str
    category: str = "mmp"
    rule_source: str = "matched_molecular_pair"
    reaction_family: str = "matched_molecular_pair"
    synthetic_route: str | None = None
    synthetic_feasibility_score: float = 0.50
    medchem_realism_score: float = 0.50
    transformation_confidence: float = 0.50
    preserves_scaffold: bool = True


MMP_RULES = [
    TransformationRule("anisole_to_phenetole", "[c:1][O:2][CH3]>>[c:1][O:2]CC", "late_stage_alkoxy_tuning", 0.76, 0.74, 0.72),
    TransformationRule("phenetole_to_anisole", "[c:1][O:2]CC>>[c:1][O:2]C", "late_stage_alkoxy_tuning", 0.78, 0.76, 0.74),
    TransformationRule("anisole_to_phenol", "[c:1][O:2][CH3]>>[c:1][O:2]", "demethylation", 0.82, 0.77, 0.75),
    TransformationRule("phenol_to_anisole", "[c:1][O:2]>>[c:1][O:2]C", "methylation", 0.84, 0.80, 0.78),
    TransformationRule("aryl_cl_to_f", "[c:1][Cl]>>[c:1]F", "halogen_scan", 0.84, 0.81, 0.78),
    TransformationRule("aryl_f_to_cl", "[c:1][F]>>[c:1]Cl", "halogen_scan", 0.82, 0.79, 0.76),
    TransformationRule("aryl_br_to_cl", "[c:1][Br]>>[c:1]Cl", "halogen_scan", 0.80, 0.77, 0.74),
    TransformationRule("aryl_cl_to_cn", "[c:1][Cl]>>[c:1]C#N", "halogen_to_nitrile", 0.77, 0.75, 0.74),
    TransformationRule("aryl_f_to_cn", "[c:1][F]>>[c:1]C#N", "halogen_to_nitrile", 0.79, 0.77, 0.76),
    TransformationRule("terminal_alkyne_to_nitrile", "[*:1]C#C>>[*:1]C#N", "bioisostere_swap", 0.74, 0.73, 0.72),
    TransformationRule("nitrile_to_terminal_alkyne", "[*:1]C#N>>[*:1]C#C", "bioisostere_swap", 0.72, 0.70, 0.68),
    TransformationRule("hydroxymethyl_to_fluoromethyl", "[*:1]CO>>[*:1]CF", "bioisostere_swap", 0.73, 0.72, 0.70),
    TransformationRule("fluoromethyl_to_hydroxymethyl", "[*:1]CF>>[*:1]CO", "bioisostere_swap", 0.75, 0.74, 0.72),
    TransformationRule("methoxy_to_trifluoromethoxy", "[c:1]OC>>[c:1]OC(F)(F)F", "alkoxy_tail_scan", 0.70, 0.68, 0.66),
    TransformationRule("trifluoromethoxy_to_methoxy", "[c:1]OC(F)(F)F>>[c:1]OC", "alkoxy_tail_scan", 0.71, 0.69, 0.67),
    TransformationRule("morpholine_to_piperazine", "[N:1]1CCOCC1>>[N:1]1CCNCC1", "ring_swap", 0.71, 0.69, 0.67),
    TransformationRule("piperazine_to_morpholine", "[N:1]1CCNCC1>>[N:1]1CCOCC1", "ring_swap", 0.73, 0.71, 0.69),
    TransformationRule("piperidine_to_morpholine", "[*:1]N1CCCCC1>>[*:1]N1CCOCC1", "solubilizing_ring_swap", 0.69, 0.67, 0.64),
    TransformationRule("morpholine_to_piperidine", "[*:1]N1CCOCC1>>[*:1]N1CCCCC1", "solubilizing_ring_swap", 0.68, 0.66, 0.63),
]


def _compile_rule(rule: TransformationRule):
    with rdBase.BlockLogs():
        return AllChem.ReactionFromSmarts(rule.reaction_smarts)


def _same_core_scaffold(parent_smiles: str, candidate_smiles: str) -> bool:
    parent_scaffold = murcko_scaffold_smiles(parent_smiles)
    candidate_scaffold = murcko_scaffold_smiles(candidate_smiles)
    return parent_scaffold == candidate_scaffold


def _sanitize_candidate(mol) -> str | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def generate_mmp_variants(smiles: str, max_variants: int = 80) -> list[str]:
    return [outcome.smiles for outcome in generate_mmp_outcomes(smiles, max_variants=max_variants)]


def generate_mmp_outcomes(smiles: str, max_variants: int = 80) -> list[TransformationOutcome]:
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []

    outcomes: dict[str, TransformationOutcome] = {}
    for rule in MMP_RULES:
        rxn = _compile_rule(rule)
        with rdBase.BlockLogs():
            products = rxn.RunReactants((parent,))
        for product_tuple in products:
            if not product_tuple:
                continue
            candidate = _sanitize_candidate(product_tuple[0])
            if not candidate or candidate == smiles:
                continue
            if not _same_core_scaffold(smiles, candidate):
                continue
            outcomes[candidate] = TransformationOutcome(
                action_name=rule.name,
                smiles=candidate,
                synthetic_route=rule.synthetic_route,
                synthetic_feasibility_score=rule.synthetic_feasibility_score,
                medchem_realism_score=rule.medchem_realism_score,
                transformation_confidence=rule.transformation_confidence,
                preserves_scaffold=True,
            )
            if len(outcomes) >= max_variants:
                return sorted(outcomes.values(), key=lambda item: item.smiles)
    return sorted(outcomes.values(), key=lambda item: item.smiles)
