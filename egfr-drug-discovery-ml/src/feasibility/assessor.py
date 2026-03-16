from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, Crippen, Descriptors, QED, rdMolDescriptors

from src.config import PROJECT_ROOT
from src.data.dataset_registry import resolve_preferred_processed_dataset
from src.utils.advanced_filters import pains_alert, severe_structural_alerts, structural_alerts
from src.utils.sa_score import simple_sa_score
from src.utils.similarity import bulk_tanimoto_similarity, morgan_fp, murcko_scaffold_smiles


@dataclass(frozen=True)
class FeasibilityContext:
    active_smiles: list[str]
    active_fps: list[Any]
    active_scaffolds: set[str]
    marketed_scaffolds: set[str]
    fragment_counts: Counter
    active_source_counts: dict[str, int]
    n_reference_molecules: int


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _brics_fragments(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {fragment for fragment in BRICS.BRICSDecompose(mol) if fragment}


@lru_cache(maxsize=2)
def _load_context(active_threshold: float = 8.0) -> FeasibilityContext:
    dataset_path = resolve_preferred_processed_dataset()
    df = pd.read_csv(dataset_path, low_memory=False)
    if "smiles_canonical" not in df.columns or "pIC50_median" not in df.columns:
        raise ValueError(f"Processed dataset missing required columns: {dataset_path}")

    active_df = df[df["pIC50_median"] >= active_threshold].copy()
    marketed_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
    marketed_df = pd.read_csv(marketed_path, low_memory=False) if marketed_path.exists() else pd.DataFrame(columns=["smiles"])

    reference_smiles = pd.concat(
        [
            active_df["smiles_canonical"].dropna().astype(str),
            marketed_df.get("smiles", pd.Series(dtype=str)).dropna().astype(str),
        ],
        ignore_index=True,
    ).drop_duplicates()

    active_smiles: list[str] = []
    active_fps: list[Any] = []
    active_scaffolds: set[str] = set()
    marketed_scaffolds: set[str] = set()
    fragment_counts: Counter = Counter()
    active_source_counts: dict[str, int] = {}

    for _, row in active_df.iterrows():
        smiles = str(row.get("smiles_canonical", "") or "")
        if not smiles:
            continue
        try:
            source_count = int(row.get("n_sources", 1) or 1)
        except (TypeError, ValueError):
            source_count = 1
        active_source_counts[smiles] = max(active_source_counts.get(smiles, 1), max(1, source_count))

    for smiles in reference_smiles.tolist():
        fp = morgan_fp(smiles=smiles)
        if fp is None:
            continue
        active_smiles.append(smiles)
        active_fps.append(fp)
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold:
            active_scaffolds.add(scaffold)
        for fragment in _brics_fragments(smiles):
            fragment_counts[fragment] += 1

    for smiles in marketed_df.get("smiles", pd.Series(dtype=str)).dropna().astype(str).tolist():
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold:
            marketed_scaffolds.add(scaffold)

    return FeasibilityContext(
        active_smiles=active_smiles,
        active_fps=active_fps,
        active_scaffolds=active_scaffolds,
        marketed_scaffolds=marketed_scaffolds,
        fragment_counts=fragment_counts,
        active_source_counts=active_source_counts,
        n_reference_molecules=len(active_smiles),
    )


class FeasibilityAssessor:
    def __init__(self, active_threshold: float = 8.0):
        self.context = _load_context(active_threshold)

    def assess(
        self,
        smiles: str,
        parent_smiles: str | None = None,
        action_name: str | None = None,
        synthetic_feasibility_score: float | None = None,
        medchem_realism_score: float | None = None,
        transformation_confidence: float | None = None,
        reaction_family: str | None = None,
        docking_rescore: float | None = None,
        interaction_support_score: float | None = None,
        interaction_key_residue_count: int | None = None,
    ) -> dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for feasibility assessment: {smiles}")

        fp = morgan_fp(mol=mol)
        if fp is None:
            raise ValueError(f"Unable to fingerprint SMILES: {smiles}")

        active_sims = bulk_tanimoto_similarity(fp, self.context.active_fps)
        max_active_similarity = max(active_sims) if active_sims else 0.0
        mean_top5_active_similarity = sum(sorted(active_sims, reverse=True)[:5]) / max(1, min(5, len(active_sims))) if active_sims else 0.0
        active_neighbor_count = int(sum(1 for sim in active_sims if sim >= 0.55))
        high_conf_neighbor_sources = [
            self.context.active_source_counts.get(ref_smiles, 1)
            for sim, ref_smiles in zip(active_sims, self.context.active_smiles)
            if sim >= 0.55
        ]
        mean_neighbor_sources = (
            sum(high_conf_neighbor_sources) / len(high_conf_neighbor_sources)
            if high_conf_neighbor_sources
            else 1.0
        )
        max_neighbor_sources = max(high_conf_neighbor_sources) if high_conf_neighbor_sources else 1
        source_support_score = _clip01((mean_neighbor_sources - 1.0) / 2.0)

        scaffold = murcko_scaffold_smiles(smiles) or ""
        scaffold_in_active_set = scaffold in self.context.active_scaffolds
        scaffold_in_marketed_set = scaffold in self.context.marketed_scaffolds

        fragment_set = _brics_fragments(smiles)
        supported_fragments = sum(1 for fragment in fragment_set if self.context.fragment_counts.get(fragment, 0) > 0)
        fragment_support_ratio = supported_fragments / max(1, len(fragment_set))
        fragment_frequency_score = 0.0
        if fragment_set:
            fragment_frequency_score = sum(
                min(1.0, self.context.fragment_counts.get(fragment, 0) / 5.0)
                for fragment in fragment_set
            ) / len(fragment_set)

        parent_similarity = None
        if parent_smiles:
            parent_fp = morgan_fp(smiles=parent_smiles)
            if parent_fp is not None:
                parent_similarity = float(DataStructs.TanimotoSimilarity(fp, parent_fp))

        sa = float(simple_sa_score(smiles) or 10.0)
        qed = float(QED.qed(mol))
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        lipinski_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

        has_pains, _ = pains_alert(smiles)
        alert_count = len(structural_alerts(smiles))
        severe_alert_count = len(severe_structural_alerts(smiles))

        active_support_score = _clip01((mean_top5_active_similarity - 0.30) / 0.50)
        scaffold_support_score = 1.0 if (scaffold_in_active_set or scaffold_in_marketed_set) else 0.0
        traceability_score = 1.0 if action_name else 0.0
        synthetic_ease_score = _clip01(1.0 - (sa / 6.5))
        route_synthetic_support = _clip01(0.60 if synthetic_feasibility_score is None else float(synthetic_feasibility_score))
        medchem_realism_support = _clip01(0.55 if medchem_realism_score is None else float(medchem_realism_score))
        transformation_confidence_support = _clip01(0.55 if transformation_confidence is None else float(transformation_confidence))
        lipinski_support_score = _clip01(1.0 - (lipinski_violations / 4.0))
        structural_support_score = 0.5 if docking_rescore is None else _clip01(float(docking_rescore))
        interaction_support = 0.0 if interaction_support_score is None else _clip01(float(interaction_support_score))
        interaction_key_support = 0.0 if interaction_key_residue_count is None else _clip01(float(interaction_key_residue_count) / 4.0)

        feasibility_score = (
            0.18 * active_support_score
            + 0.16 * fragment_support_ratio
            + 0.08 * fragment_frequency_score
            + 0.11 * scaffold_support_score
            + 0.08 * traceability_score
            + 0.08 * synthetic_ease_score
            + 0.07 * route_synthetic_support
            + 0.06 * medchem_realism_support
            + 0.05 * transformation_confidence_support
            + 0.07 * lipinski_support_score
            + 0.05 * qed
            + 0.03 * structural_support_score
            + 0.09 * interaction_support
            + 0.04 * interaction_key_support
        )

        risk_penalty = (
            0.10 * int(has_pains)
            + 0.05 * alert_count
            + 0.10 * severe_alert_count
        )
        feasibility_score = _clip01(feasibility_score - risk_penalty)

        evidence_bits = []
        if max_active_similarity >= 0.55:
            evidence_bits.append("near_known_active")
        if scaffold_support_score > 0:
            evidence_bits.append("known_scaffold_support")
        if fragment_support_ratio >= 0.60:
            evidence_bits.append("fragment_support")
        if traceability_score > 0:
            evidence_bits.append("traceable_generation")
        if synthetic_ease_score >= 0.40:
            evidence_bits.append("synthetic_accessibility_ok")
        if route_synthetic_support >= 0.60:
            evidence_bits.append("reaction_route_supported")
        if medchem_realism_support >= 0.62:
            evidence_bits.append("medchem_realism")
        if transformation_confidence_support >= 0.60:
            evidence_bits.append("transformation_confident")
        if source_support_score >= 0.35:
            evidence_bits.append("multi_source_neighbor_support")
        if docking_rescore is not None and docking_rescore >= 0.55:
            evidence_bits.append("structural_support")
        if interaction_support >= 0.35:
            evidence_bits.append("interaction_support")
        if interaction_key_support >= 0.25:
            evidence_bits.append("key_residue_contact")

        feasibility_status = "pass"
        if severe_alert_count >= 1 or has_pains or feasibility_score < 0.35:
            feasibility_status = "fail"
        elif feasibility_score < 0.60 or len(evidence_bits) < 3:
            feasibility_status = "review"

        return {
            "feasibility_score": feasibility_score,
            "feasibility_status": feasibility_status,
            "max_active_similarity": max_active_similarity,
            "mean_top5_active_similarity": mean_top5_active_similarity,
            "active_neighbor_count_055": active_neighbor_count,
            "mean_neighbor_sources": mean_neighbor_sources,
            "max_neighbor_sources": max_neighbor_sources,
            "source_support_score": source_support_score,
            "scaffold_in_active_set": bool(scaffold_in_active_set),
            "scaffold_in_marketed_set": bool(scaffold_in_marketed_set),
            "fragment_support_ratio": fragment_support_ratio,
            "fragment_frequency_score": fragment_frequency_score,
            "traceability_score": traceability_score,
            "parent_similarity": parent_similarity,
            "synthetic_ease_score": synthetic_ease_score,
            "route_synthetic_support_score": route_synthetic_support,
            "medchem_realism_score": medchem_realism_support,
            "transformation_confidence_score": transformation_confidence_support,
            "reaction_family": reaction_family,
            "lipinski_support_score": lipinski_support_score,
            "structural_support_score": structural_support_score,
            "interaction_support_score": interaction_support,
            "interaction_key_residue_support": interaction_key_support,
            "feasibility_evidence_count": len(evidence_bits),
            "feasibility_evidence": ";".join(evidence_bits) if evidence_bits else None,
            "feasibility_reference_count": self.context.n_reference_molecules,
        }
