from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from src.utils.advanced_filters import (
    covalent_warhead_alerts,
    pains_alert,
    severe_structural_alerts,
    structural_alerts,
)
from src.utils.sa_score import simple_sa_score
from src.utils.similarity import morgan_fp, murcko_scaffold_smiles

if TYPE_CHECKING:
    from src.generation.medchem_mutations import MutationOutcome


CATEGORY_PRIORITY = {
    "snar": 0.92,
    "acylation": 0.90,
    "alkylation": 0.83,
    "functional_group_swap": 0.80,
    "mmp": 0.86,
    "atom_swap": 0.72,
    "hetero_edit": 0.78,
    "append_group": 0.58,
}

RULE_SOURCE_PRIORITY = {
    "reaction_transform": 0.90,
    "matched_molecular_pair": 0.86,
    "hetero_edit": 0.76,
    "atom_edit": 0.72,
    "append_group": 0.56,
    "medchem_edit": 0.68,
}

HARD_ALERT_ALLOWLIST = {"aniline"}


@dataclass(frozen=True)
class GeneratorCandidateAssessment:
    hard_constraint_pass: bool
    hard_constraint_notes: str | None
    parent_similarity: float
    property_support_score: float
    category_priority_score: float
    generator_priority_score: float
    introduced_warhead: bool
    warhead_retained: bool
    alert_count: int
    severe_alert_count: int


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _range_support(value: float, ideal_low: float, ideal_high: float, floor_low: float, floor_high: float) -> float:
    if value < floor_low or value > floor_high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return _clip01((value - floor_low) / max(ideal_low - floor_low, 1e-6))
    return _clip01((floor_high - value) / max(floor_high - ideal_high, 1e-6))


def _default_quota(max_variants: int, category_count: int) -> int:
    return max(2, min(max_variants, max(3, int(round(max_variants / max(category_count, 4))))))


def assess_generator_candidate(parent_smiles: str, outcome: MutationOutcome) -> GeneratorCandidateAssessment:
    mol = Chem.MolFromSmiles(outcome.smiles)
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None or parent_mol is None:
        return GeneratorCandidateAssessment(
            hard_constraint_pass=False,
            hard_constraint_notes="invalid_smiles",
            parent_similarity=0.0,
            property_support_score=0.0,
            category_priority_score=0.0,
            generator_priority_score=0.0,
            introduced_warhead=False,
            warhead_retained=False,
            alert_count=0,
            severe_alert_count=0,
        )

    candidate_fp = morgan_fp(mol=mol)
    parent_fp = morgan_fp(mol=parent_mol)
    parent_similarity = 0.0 if candidate_fp is None or parent_fp is None else float(DataStructs.TanimotoSimilarity(candidate_fp, parent_fp))

    candidate_scaffold = murcko_scaffold_smiles(outcome.smiles)
    parent_scaffold = murcko_scaffold_smiles(parent_smiles)
    preserves_scaffold = bool(outcome.preserves_scaffold) and bool(candidate_scaffold) and candidate_scaffold == parent_scaffold

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    qed = float(QED.qed(mol))
    sa = float(simple_sa_score(outcome.smiles) or 10.0)
    lipinski_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

    has_pains, _ = pains_alert(outcome.smiles)
    alerts = structural_alerts(outcome.smiles)
    severe_alerts = [alert for alert in severe_structural_alerts(outcome.smiles) if alert not in HARD_ALERT_ALLOWLIST]
    parent_warheads = set(covalent_warhead_alerts(parent_smiles))
    candidate_warheads = set(covalent_warhead_alerts(outcome.smiles))
    introduced_warhead = bool(candidate_warheads - parent_warheads)
    warhead_retained = bool(candidate_warheads & parent_warheads) or not candidate_warheads

    property_support = (
        0.24 * qed
        + 0.17 * _clip01(1.0 - (sa / 6.2))
        + 0.14 * _range_support(mw, ideal_low=260.0, ideal_high=560.0, floor_low=80.0, floor_high=700.0)
        + 0.11 * _range_support(logp, ideal_low=1.0, ideal_high=4.8, floor_low=-0.5, floor_high=6.8)
        + 0.10 * _range_support(tpsa, ideal_low=50.0, ideal_high=120.0, floor_low=20.0, floor_high=160.0)
        + 0.08 * _clip01(1.0 - (lipinski_violations / 4.0))
        + 0.08 * _clip01(1.0 - (rot_bonds / 12.0))
        + 0.04 * _clip01(ring_count / 5.0)
        + 0.04 * _clip01(fraction_csp3 / 0.45)
    )

    category_priority = 0.65 * CATEGORY_PRIORITY.get(outcome.category, 0.66) + 0.35 * RULE_SOURCE_PRIORITY.get(outcome.rule_source, 0.66)
    realism_score = (
        0.33 * _clip01(float(outcome.synthetic_feasibility_score))
        + 0.27 * _clip01(float(outcome.medchem_realism_score))
        + 0.20 * _clip01(float(outcome.transformation_confidence))
        + 0.12 * parent_similarity
        + 0.08 * float(preserves_scaffold)
    )

    generator_priority = (
        0.34 * realism_score
        + 0.30 * property_support
        + 0.14 * category_priority
        + 0.12 * parent_similarity
        + 0.10 * float(warhead_retained)
        - 0.10 * float(introduced_warhead)
        - 0.08 * min(len(alerts), 3)
    )
    generator_priority = _clip01(generator_priority)

    fail_reasons: list[str] = []
    if not preserves_scaffold:
        fail_reasons.append("scaffold_shift")
    if has_pains:
        fail_reasons.append("pains_alert")
    if severe_alerts:
        fail_reasons.append("severe_structural_alert")
    if len(alerts) >= 4:
        fail_reasons.append("too_many_alerts")
    if qed < 0.20:
        fail_reasons.append("very_low_qed")
    if sa > 6.4:
        fail_reasons.append("high_sa")
    if parent_similarity < 0.28:
        fail_reasons.append("low_parent_similarity")
    if lipinski_violations >= 3:
        fail_reasons.append("excessive_lipinski_violations")
    if mw > 700 or mw < 80:
        fail_reasons.append("mw_out_of_range")
    if logp > 6.8:
        fail_reasons.append("logp_out_of_range")
    if rot_bonds > 14:
        fail_reasons.append("too_flexible")

    return GeneratorCandidateAssessment(
        hard_constraint_pass=not fail_reasons,
        hard_constraint_notes=";".join(fail_reasons) if fail_reasons else None,
        parent_similarity=parent_similarity,
        property_support_score=property_support,
        category_priority_score=category_priority,
        generator_priority_score=generator_priority,
        introduced_warhead=introduced_warhead,
        warhead_retained=warhead_retained,
        alert_count=len(alerts),
        severe_alert_count=len(severe_alerts),
    )


def apply_generator_policy(
    parent_smiles: str,
    outcomes: Iterable[MutationOutcome],
    max_variants: int = 100,
    max_per_category: int | None = None,
) -> list[MutationOutcome]:
    assessed: list[MutationOutcome] = []
    for outcome in outcomes:
        assessment = assess_generator_candidate(parent_smiles, outcome)
        enriched = replace(
            outcome,
            parent_similarity=assessment.parent_similarity,
            property_support_score=assessment.property_support_score,
            category_priority_score=assessment.category_priority_score,
            generator_priority_score=assessment.generator_priority_score,
            hard_constraint_pass=assessment.hard_constraint_pass,
            hard_constraint_notes=assessment.hard_constraint_notes,
            introduced_warhead=assessment.introduced_warhead,
            warhead_retained=assessment.warhead_retained,
            alert_count=assessment.alert_count,
            severe_alert_count=assessment.severe_alert_count,
        )
        if assessment.hard_constraint_pass:
            assessed.append(enriched)

    if not assessed:
        return []

    assessed.sort(
        key=lambda item: (
            float(item.generator_priority_score),
            float(item.synthetic_feasibility_score),
            float(item.medchem_realism_score),
            float(item.transformation_confidence),
            float(item.parent_similarity),
        ),
        reverse=True,
    )

    categories = {item.category for item in assessed}
    per_category_quota = max_per_category or _default_quota(max_variants=max_variants, category_count=len(categories))
    selected: list[MutationOutcome] = []
    category_counts: dict[str, int] = defaultdict(int)

    for item in assessed:
        if len(selected) >= max_variants:
            break
        if category_counts[item.category] >= per_category_quota:
            continue
        selected.append(item)
        category_counts[item.category] += 1

    if len(selected) < max_variants:
        seen = {item.smiles for item in selected}
        for item in assessed:
            if len(selected) >= max_variants:
                break
            if item.smiles in seen:
                continue
            selected.append(item)
            seen.add(item.smiles)

    return selected[:max_variants]
