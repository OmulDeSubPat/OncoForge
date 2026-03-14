from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from src.utils.similarity import mol_from_smiles, murcko_scaffold_smiles


@dataclass(frozen=True)
class TransformationRule:
    name: str
    reaction_smarts: str


MMP_RULES = [
    TransformationRule("anisole_to_phenetole", "[c:1][O:2][CH3]>>[c:1][O:2]CC"),
    TransformationRule("phenetole_to_anisole", "[c:1][O:2]CC>>[c:1][O:2]C"),
    TransformationRule("aryl_cl_to_f", "[c:1][Cl]>>[c:1]F"),
    TransformationRule("aryl_f_to_cl", "[c:1][F]>>[c:1]Cl"),
    TransformationRule("aryl_br_to_cl", "[c:1][Br]>>[c:1]Cl"),
    TransformationRule("terminal_alkyne_to_nitrile", "[*:1]C#C>>[*:1]C#N"),
    TransformationRule("nitrile_to_terminal_alkyne", "[*:1]C#N>>[*:1]C#C"),
    TransformationRule("hydroxymethyl_to_fluoromethyl", "[*:1]CO>>[*:1]CF"),
    TransformationRule("fluoromethyl_to_hydroxymethyl", "[*:1]CF>>[*:1]CO"),
    TransformationRule("morpholine_to_piperazine", "[N:1]1CCOCC1>>[N:1]1CCNCC1"),
    TransformationRule("piperazine_to_morpholine", "[N:1]1CCNCC1>>[N:1]1CCOCC1"),
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
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []

    variants: set[str] = set()
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
            variants.add(candidate)
            if len(variants) >= max_variants:
                return sorted(variants)
    return sorted(variants)
