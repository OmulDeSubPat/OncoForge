from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.data.fetch_excape_egfr import ensure_excape_reference
from src.data.fetch_iuphar_egfr import ensure_iuphar_reference
from src.data.fetch_papyrus_egfr import ensure_papyrus_reference
from src.data.fetch_pubchem_egfr import ensure_pubchem_reference
from src.utils.similarity import bulk_tanimoto_similarity, morgan_fp, murcko_scaffold_smiles


@dataclass(frozen=True)
class ValidationReferenceLibrary:
    source_name: str
    smiles: list[str]
    fps: list[Any]
    labels: list[str]
    scaffolds: set[str]
    activity_values: list[float]
    evidence_values: list[float]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_label(value: Any, fallback: str) -> str:
    if pd.isna(value) or value in (None, ""):
        return fallback
    return str(value)


def _build_library(
    df: pd.DataFrame,
    source_name: str,
    smiles_col: str,
    label_col: str,
    activity_col: str,
    evidence_col: str | None = None,
) -> ValidationReferenceLibrary:
    smiles_list: list[str] = []
    fps: list[Any] = []
    labels: list[str] = []
    scaffolds: set[str] = set()
    activities: list[float] = []
    evidence_values: list[float] = []

    for _, row in df.iterrows():
        smiles = row.get(smiles_col)
        fp = morgan_fp(smiles=smiles)
        if fp is None:
            continue
        smiles = str(smiles)
        smiles_list.append(smiles)
        fps.append(fp)
        labels.append(_safe_label(row.get(label_col), smiles))
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold:
            scaffolds.add(scaffold)
        activities.append(float(pd.to_numeric(row.get(activity_col), errors="coerce")) if pd.notna(row.get(activity_col)) else np.nan)
        if evidence_col:
            evidence_values.append(float(pd.to_numeric(row.get(evidence_col), errors="coerce")) if pd.notna(row.get(evidence_col)) else np.nan)
        else:
            evidence_values.append(np.nan)

    return ValidationReferenceLibrary(
        source_name=source_name,
        smiles=smiles_list,
        fps=fps,
        labels=labels,
        scaffolds=scaffolds,
        activity_values=activities,
        evidence_values=evidence_values,
    )


@lru_cache(maxsize=1)
def load_reference_libraries(active_threshold: float = 7.0) -> dict[str, ValidationReferenceLibrary]:
    interim_path = INTERIM_DIR / "egfr_multisource_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "egfr_multisource_ic50_clean.csv"
    if not interim_path.exists() or not processed_path.exists():
        raise FileNotFoundError("Missing multisource EGFR artifacts required for cross-database validation.")

    interim = pd.read_csv(interim_path, low_memory=False)
    processed = pd.read_csv(processed_path, low_memory=False)

    chembl_df = interim[
        (interim["source_dataset"].astype(str) == "chembl")
        & (pd.to_numeric(interim["pIC50"], errors="coerce") >= active_threshold)
    ].copy()
    bindingdb_df = interim[
        (interim["source_dataset"].astype(str).str.contains("bindingdb"))
        & (pd.to_numeric(interim["pIC50"], errors="coerce") >= active_threshold)
    ].copy()
    multisource_df = processed[
        (pd.to_numeric(processed["pIC50_median"], errors="coerce") >= active_threshold)
        & (pd.to_numeric(processed["n_sources"], errors="coerce") >= 2)
    ].copy()

    iuphar_path = ensure_iuphar_reference()
    iuphar_df = pd.read_csv(iuphar_path, low_memory=False)
    iuphar_df = iuphar_df[pd.to_numeric(iuphar_df["pIC50_median"], errors="coerce") >= active_threshold].copy()

    papyrus_path = ensure_papyrus_reference()
    papyrus_df = pd.read_csv(papyrus_path, low_memory=False)
    papyrus_df = papyrus_df[
        (pd.to_numeric(papyrus_df["pIC50_median"], errors="coerce") >= active_threshold)
        & (pd.to_numeric(papyrus_df.get("papyrus_support_score", 0.0), errors="coerce") >= 0.45)
    ].copy()

    excape_path = ensure_excape_reference()
    excape_df = pd.read_csv(excape_path, low_memory=False)
    excape_df = excape_df[
        (pd.to_numeric(excape_df["pIC50_median"], errors="coerce") >= active_threshold)
        & (pd.to_numeric(excape_df.get("excape_support_score", 0.0), errors="coerce") >= 0.40)
        & (pd.to_numeric(excape_df.get("active_fraction", 0.0), errors="coerce") >= 0.30)
    ].copy()

    pubchem_path = ensure_pubchem_reference()
    pubchem_df = pd.read_csv(pubchem_path, low_memory=False)
    pubchem_active_mask = pubchem_df["consensus_active"].astype(str).str.lower().isin(["true", "1", "yes"])
    pubchem_evidence_col = "pubchem_enriched_evidence_score" if "pubchem_enriched_evidence_score" in pubchem_df.columns else "pubchem_evidence_score"
    pubchem_df = pubchem_df[
        pubchem_active_mask
        & (pd.to_numeric(pubchem_df[pubchem_evidence_col], errors="coerce") >= 0.45)
        & (pd.to_numeric(pubchem_df.get("virtual_proxy_fraction", 0.0), errors="coerce").fillna(0.0) <= 0.80)
    ].copy()

    marketed_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
    marketed_df = pd.read_csv(marketed_path, low_memory=False) if marketed_path.exists() else pd.DataFrame(columns=["name", "smiles"])
    marketed_df["pIC50_proxy"] = np.nan

    libraries = {
        "chembl": _build_library(chembl_df, "chembl", "smiles_canonical", "molecule_chembl_id", "pIC50"),
        "bindingdb": _build_library(bindingdb_df, "bindingdb", "smiles_canonical", "bindingdb_reactant_set_id", "pIC50"),
        "multisource_consensus": _build_library(multisource_df, "multisource_consensus", "smiles_canonical", "source_datasets", "pIC50_median"),
        "iuphar": _build_library(iuphar_df, "iuphar", "smiles", "ligand_name", "pIC50_median"),
        "papyrus": _build_library(papyrus_df, "papyrus", "smiles_canonical", "papyrus_sources", "pIC50_median", evidence_col="papyrus_support_score"),
        "excape": _build_library(excape_df, "excape", "smiles_canonical", "source_dbs", "pIC50_median", evidence_col="excape_support_score"),
        "pubchem": _build_library(pubchem_df, "pubchem", "smiles", "CID", "best_proxy_pIC50", evidence_col=pubchem_evidence_col),
        "marketed": _build_library(marketed_df, "marketed", "smiles", "name", "pIC50_proxy"),
    }
    return libraries


class CrossDatabaseValidator:
    def __init__(self, active_threshold: float = 7.0):
        self.libraries = load_reference_libraries(active_threshold)

    def _validate_smiles(self, smiles: str) -> dict[str, Any]:
        fp = morgan_fp(smiles=smiles)
        if fp is None:
            raise ValueError(f"Invalid SMILES for cross-database validation: {smiles}")

        scaffold = murcko_scaffold_smiles(smiles) or ""
        source_support_scores: dict[str, float] = {}
        source_max_similarity: dict[str, float] = {}
        source_scaffold_hits: dict[str, bool] = {}
        source_best_match: dict[str, str | None] = {}
        source_best_match_activity: dict[str, float | None] = {}
        source_best_match_evidence: dict[str, float | None] = {}
        supporting_sources: list[str] = []

        for source_name, library in self.libraries.items():
            sims = bulk_tanimoto_similarity(fp, library.fps)
            max_similarity = max(sims) if sims else 0.0
            source_max_similarity[source_name] = float(max_similarity)
            scaffold_hit = bool(scaffold and scaffold in library.scaffolds)
            source_scaffold_hits[source_name] = scaffold_hit
            best_evidence = np.nan
            if sims:
                best_idx = int(np.argmax(sims))
                best_evidence = library.evidence_values[best_idx]
            evidence_bonus = 0.0 if np.isnan(best_evidence) else float(best_evidence)
            if source_name == "pubchem":
                support_score = _clip01(0.62 * max_similarity + 0.12 * float(scaffold_hit) + 0.26 * evidence_bonus)
            elif source_name in {"papyrus", "excape"}:
                support_score = _clip01(0.68 * max_similarity + 0.12 * float(scaffold_hit) + 0.20 * evidence_bonus)
            else:
                support_score = _clip01(0.82 * max_similarity + 0.18 * float(scaffold_hit))
            source_support_scores[source_name] = support_score

            if sims:
                source_best_match[source_name] = library.labels[best_idx]
                activity_value = library.activity_values[best_idx]
                source_best_match_activity[source_name] = None if np.isnan(activity_value) else float(activity_value)
                source_best_match_evidence[source_name] = None if np.isnan(best_evidence) else float(best_evidence)
            else:
                source_best_match[source_name] = None
                source_best_match_activity[source_name] = None
                source_best_match_evidence[source_name] = None

            if support_score >= 0.55 or scaffold_hit:
                supporting_sources.append(source_name)

        independent_sources = ["chembl", "bindingdb", "iuphar", "pubchem"]
        secondary_sources = ["papyrus", "excape"]
        independent_support_count = int(sum(source_support_scores[source] >= 0.55 or source_scaffold_hits[source] for source in independent_sources))
        secondary_support_count = int(sum(source_support_scores[source] >= 0.55 or source_scaffold_hits[source] for source in secondary_sources))
        external_support_count = int(sum(source_support_scores[source] >= 0.55 for source in ["bindingdb", "iuphar", "pubchem", "papyrus", "excape"]))

        consensus_score = (
            0.20 * source_support_scores["chembl"]
            + 0.16 * source_support_scores["bindingdb"]
            + 0.14 * source_support_scores["iuphar"]
            + 0.18 * source_support_scores["pubchem"]
            + 0.12 * source_support_scores["papyrus"]
            + 0.10 * source_support_scores["excape"]
            + 0.07 * source_support_scores["multisource_consensus"]
            + 0.03 * source_support_scores["marketed"]
        )
        agreement_score = independent_support_count / max(1, len(independent_sources))
        strong = (
            (independent_support_count >= 2 and consensus_score >= 0.55)
            or (external_support_count >= 2 and consensus_score >= 0.53)
            or (secondary_support_count >= 2 and consensus_score >= 0.52)
            or (source_support_scores["pubchem"] >= 0.70 and consensus_score >= 0.56)
        )
        moderate = (
            (independent_support_count >= 1 and consensus_score >= 0.40)
            or (secondary_support_count >= 1 and consensus_score >= 0.43)
            or (source_support_scores["multisource_consensus"] >= 0.60)
        )
        status = "weak"
        if strong:
            status = "strong"
        elif moderate:
            status = "moderate"

        evidence_bits = []
        if source_support_scores["chembl"] >= 0.55:
            evidence_bits.append("chembl_support")
        if source_support_scores["bindingdb"] >= 0.55:
            evidence_bits.append("bindingdb_support")
        if source_support_scores["iuphar"] >= 0.55:
            evidence_bits.append("iuphar_support")
        if source_support_scores["papyrus"] >= 0.55:
            evidence_bits.append("papyrus_support")
        if source_support_scores["excape"] >= 0.55:
            evidence_bits.append("excape_support")
        if source_support_scores["pubchem"] >= 0.55:
            evidence_bits.append("pubchem_support")
        if source_support_scores["multisource_consensus"] >= 0.60:
            evidence_bits.append("multisource_consensus")
        if any(source_scaffold_hits.values()):
            evidence_bits.append("cross_db_scaffold_support")
        if external_support_count >= 2:
            evidence_bits.append("independent_external_agreement")

        result = {
            "chembl_max_similarity": source_max_similarity["chembl"],
            "bindingdb_max_similarity": source_max_similarity["bindingdb"],
            "iuphar_max_similarity": source_max_similarity["iuphar"],
            "multisource_max_similarity": source_max_similarity["multisource_consensus"],
            "cross_database_consensus_score": float(consensus_score),
            "cross_database_agreement_score": float(agreement_score),
            "cross_database_independent_support_count": independent_support_count,
            "cross_database_secondary_support_count": secondary_support_count,
            "cross_database_external_support_count": external_support_count,
            "cross_database_supporting_sources": ";".join(sorted(supporting_sources)) if supporting_sources else None,
            "cross_database_status": status,
            "cross_database_evidence": ";".join(evidence_bits) if evidence_bits else None,
            "cross_database_evidence_count": len(evidence_bits),
        }

        for source_name in self.libraries:
            result[f"{source_name}_support_score"] = source_support_scores[source_name]
            result[f"{source_name}_scaffold_hit"] = bool(source_scaffold_hits[source_name])
            result[f"{source_name}_best_match"] = source_best_match[source_name]
            result[f"{source_name}_best_match_activity"] = source_best_match_activity[source_name]
            result[f"{source_name}_best_match_evidence"] = source_best_match_evidence[source_name]

        return result

    def validate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        rows = []
        for _, row in df.iterrows():
            out_row = row.to_dict()
            try:
                out_row.update(self._validate_smiles(str(row["smiles"])))
            except Exception:
                out_row.update(
                    {
                        "chembl_max_similarity": np.nan,
                        "bindingdb_max_similarity": np.nan,
                        "iuphar_max_similarity": np.nan,
                        "pubchem_max_similarity": np.nan,
                        "multisource_max_similarity": np.nan,
                        "cross_database_consensus_score": 0.0,
                        "cross_database_agreement_score": 0.0,
                        "cross_database_independent_support_count": 0,
                        "cross_database_secondary_support_count": 0,
                        "cross_database_external_support_count": 0,
                        "cross_database_supporting_sources": None,
                        "cross_database_status": "weak",
                        "cross_database_evidence": None,
                        "cross_database_evidence_count": 0,
                    }
                )
            rows.append(out_row)

        out = pd.DataFrame(rows)
        if "experimental_readiness_priority" in out.columns:
            out["cross_database_priority"] = (
                pd.to_numeric(out["experimental_readiness_priority"], errors="coerce").fillna(0.0)
                + 0.85 * pd.to_numeric(out["cross_database_consensus_score"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(out["cross_database_independent_support_count"], errors="coerce").fillna(0.0)
            )
        elif "final_score" in out.columns:
            out["cross_database_priority"] = (
                pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
                + 0.85 * pd.to_numeric(out["cross_database_consensus_score"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(out["cross_database_independent_support_count"], errors="coerce").fillna(0.0)
            )
        else:
            out["cross_database_priority"] = (
                pd.to_numeric(out["cross_database_consensus_score"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(out["cross_database_independent_support_count"], errors="coerce").fillna(0.0)
            )
        return out
