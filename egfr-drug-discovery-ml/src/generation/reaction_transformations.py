from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from src.utils.similarity import mol_from_smiles, murcko_scaffold_smiles


@dataclass(frozen=True)
class ReactionTransformation:
    name: str
    reaction_smarts: str
    category: str
    synthetic_route: str
    synthetic_feasibility_score: float
    medchem_realism_score: float
    transformation_confidence: float


@dataclass(frozen=True)
class ReactionOutcome:
    action_name: str
    smiles: str
    category: str = "reaction_transform"
    rule_source: str = "reaction_transform"
    reaction_family: str = "reaction_transform"
    synthetic_route: str | None = None
    synthetic_feasibility_score: float = 0.50
    medchem_realism_score: float = 0.50
    transformation_confidence: float = 0.50
    preserves_scaffold: bool = True


REACTION_RULES = [
    ReactionTransformation("amide_acryloylation", "[N;H1:1]>>[N:1]C(=O)C=C", "acylation", "amide_coupling", 0.82, 0.72, 0.80),
    ReactionTransformation("amide_crotonylation", "[N;H1:1]>>[N:1]C(=O)C(C)=C", "acylation", "amide_coupling", 0.80, 0.70, 0.78),
    ReactionTransformation("alcohol_fluorination", "[C:1][O:2]>>[C:1]F", "functional_group_swap", "halogen_exchange", 0.74, 0.66, 0.70),
    ReactionTransformation("alcohol_chlorination", "[C:1][O:2]>>[C:1]Cl", "functional_group_swap", "halogen_exchange", 0.72, 0.64, 0.68),
    ReactionTransformation("aryl_f_to_methoxy", "[c:1]F>>[c:1]OC", "snar", "snar_diversification", 0.87, 0.82, 0.88),
    ReactionTransformation("aryl_cl_to_methoxy", "[c:1]Cl>>[c:1]OC", "snar", "snar_diversification", 0.85, 0.80, 0.86),
    ReactionTransformation("aryl_f_to_methylamino", "[c:1]F>>[c:1]NC", "snar", "snar_diversification", 0.86, 0.83, 0.87),
    ReactionTransformation("aryl_cl_to_methylamino", "[c:1]Cl>>[c:1]NC", "snar", "snar_diversification", 0.84, 0.81, 0.85),
    ReactionTransformation("aryl_f_to_cyano", "[c:1]F>>[c:1]C#N", "snar", "snar_diversification", 0.79, 0.75, 0.80),
    ReactionTransformation("aryl_cl_to_cyano", "[c:1]Cl>>[c:1]C#N", "snar", "snar_diversification", 0.77, 0.73, 0.78),
    ReactionTransformation("aryl_f_to_dimethylamino", "[c:1]F>>[c:1]N(C)C", "snar", "snar_diversification", 0.79, 0.76, 0.80),
    ReactionTransformation("aryl_cl_to_dimethylamino", "[c:1]Cl>>[c:1]N(C)C", "snar", "snar_diversification", 0.77, 0.74, 0.78),
    ReactionTransformation("methoxy_to_ethoxy", "[c:1]OC>>[c:1]OCC", "williamson", "o_alkylation", 0.83, 0.79, 0.82),
    ReactionTransformation("methoxy_to_isopropoxy", "[c:1]OC>>[c:1]OC(C)C", "williamson", "o_alkylation", 0.81, 0.76, 0.79),
    ReactionTransformation("methoxy_to_trifluoromethoxy", "[c:1]OC>>[c:1]OC(F)(F)F", "williamson", "o_alkylation", 0.73, 0.69, 0.71),
    ReactionTransformation("phenol_methylation", "[c:1][O;H1:2]>>[c:1]OC", "o_alkylation", "o_alkylation", 0.88, 0.84, 0.90),
    ReactionTransformation("phenol_ethylation", "[c:1][O;H1:2]>>[c:1]OCC", "o_alkylation", "o_alkylation", 0.85, 0.81, 0.84),
    ReactionTransformation("phenol_carbamylation", "[c:1][O;H1:2]>>[c:1]OC(=O)NC", "carbamate", "carbamate_installation", 0.79, 0.74, 0.77),
    ReactionTransformation("aniline_methylation", "[N;H1:1][c:2]>>[N:1](C)[c:2]", "alkylation", "n_alkylation", 0.84, 0.80, 0.83),
    ReactionTransformation("aniline_ethylation", "[N;H1:1][c:2]>>[N:1](CC)[c:2]", "alkylation", "n_alkylation", 0.81, 0.77, 0.80),
    ReactionTransformation("aniline_acetylation", "[N;H1:1][c:2]>>[N:1](C(=O)C)[c:2]", "acylation", "amide_coupling", 0.84, 0.79, 0.82),
    ReactionTransformation("aniline_urea_capping", "[N;H1:1][c:2]>>[N:1](C(=O)NC)[c:2]", "urea", "urea_installation", 0.83, 0.81, 0.82),
    ReactionTransformation("aniline_cyclopropyl_urea", "[N;H1:1][c:2]>>[N:1](C(=O)NC1CC1)[c:2]", "urea", "urea_installation", 0.80, 0.79, 0.80),
]


def _compile_reaction(rule: ReactionTransformation):
    with rdBase.BlockLogs():
        return AllChem.ReactionFromSmarts(rule.reaction_smarts)


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


def generate_reaction_outcomes(smiles: str, max_variants: int = 80) -> list[ReactionOutcome]:
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []

    outcomes: dict[str, ReactionOutcome] = {}
    for rule in REACTION_RULES:
        reaction = _compile_reaction(rule)
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
            outcomes[candidate] = ReactionOutcome(
                action_name=rule.name,
                smiles=candidate,
                category=rule.category,
                reaction_family=rule.category,
                synthetic_route=rule.synthetic_route,
                synthetic_feasibility_score=rule.synthetic_feasibility_score,
                medchem_realism_score=rule.medchem_realism_score,
                transformation_confidence=rule.transformation_confidence,
                preserves_scaffold=True,
            )
            if len(outcomes) >= max_variants:
                return sorted(outcomes.values(), key=lambda item: item.smiles)
    return sorted(outcomes.values(), key=lambda item: item.smiles)
