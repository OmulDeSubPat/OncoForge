from __future__ import annotations

import json
import math
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from src.utils.chem import canonicalize_smiles


PUBCHEM_DIR = EXTERNAL_DIR / "pubchem"
PUBCHEM_EGFR_CONCISE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/genesymbol/EGFR/concise/CSV"
PUBCHEM_PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_batch}/property/ConnectivitySMILES,IUPACName/CSV"
HUMAN_EGFR_GENE_ID = 1956
PROPERTY_BATCH_SIZE = 200
PUBCHEM_SCHEMA_VERSION = "2.0"
REQUIRED_PROCESSED_COLUMNS = {
    "CID",
    "smiles",
    "pubchem_evidence_score",
    "pubchem_enriched_evidence_score",
    "pubchem_relevance_score",
    "pubchem_orthogonal_support_score",
    "virtual_proxy_fraction",
    "direct_kinase_fraction",
    "mutant_fraction",
    "consensus_active",
}

DIRECT_KINASE_PATTERNS = (
    "kinase",
    "phosphorylation",
    "enzyme",
    "selectivity",
    "profiling",
    "binding",
)
CELLULAR_PATTERNS = (
    "cell",
    "cells",
    "western blot",
    "protein level",
    "degradation",
    "signaling",
    "pathway",
)
MUTANT_PATTERNS = (
    "t790m",
    "l858r",
    "c797s",
    "g719",
    "s768i",
    "l861q",
    "ex19del",
    "del19",
    "exon 19",
    "exon19",
)
VIRTUAL_PROXY_PATTERNS = (
    "beliefdocking",
    "virtual",
    "in silico",
    "computational",
    "docking",
)
PANEL_PATTERNS = (
    "kinome",
    "selectivity profiling",
    "profiling assay",
)


def _fetch_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), low_memory=False)


def _activity_um_to_pic50(value_um: float | int | None) -> float | None:
    if value_um is None:
        return None
    try:
        value = float(value_um)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return float(6.0 - math.log10(value))


def _fetch_compound_properties(cids: list[int]) -> pd.DataFrame:
    if not cids:
        return pd.DataFrame(columns=["CID", "ConnectivitySMILES", "IUPACName"])

    frames: list[pd.DataFrame] = []
    for start in range(0, len(cids), PROPERTY_BATCH_SIZE):
        batch = cids[start : start + PROPERTY_BATCH_SIZE]
        batch_str = ",".join(str(cid) for cid in batch)
        url = PUBCHEM_PROPERTY_URL.format(cid_batch=batch_str)
        frames.append(_fetch_csv(url))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["CID", "ConnectivitySMILES", "IUPACName"])


def _text_blob(*values: object) -> str:
    text = " ".join(str(value) for value in values if value not in (None, "", float("nan")))
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _flag_assay_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    assay_text = (
        out[["Activity Name", "Assay Name", "Assay Type"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out["is_confirmatory_record"] = out["assay_type"].eq("Confirmatory")
    out["is_summary_record"] = out["assay_type"].eq("Summary")
    out["is_direct_kinase_assay"] = assay_text.map(lambda value: _has_pattern(value, DIRECT_KINASE_PATTERNS))
    out["is_cellular_assay"] = assay_text.map(lambda value: _has_pattern(value, CELLULAR_PATTERNS))
    out["is_mutant_assay"] = assay_text.map(lambda value: _has_pattern(value, MUTANT_PATTERNS))
    out["is_virtual_proxy_assay"] = assay_text.map(lambda value: _has_pattern(value, VIRTUAL_PROXY_PATTERNS))
    out["is_panel_assay"] = assay_text.map(lambda value: _has_pattern(value, PANEL_PATTERNS))
    out["is_target_specific_assay"] = assay_text.str.contains("egfr|epidermal growth factor receptor", regex=True) | out["is_mutant_assay"]
    out["has_literature_support"] = out["pubmed_id"].notna()
    out["is_active_record"] = out["activity_outcome"].eq("Active")
    out["record_relevance_score"] = (
        0.24 * out["is_target_specific_assay"].astype(float)
        + 0.20 * out["is_direct_kinase_assay"].astype(float)
        + 0.12 * out["is_confirmatory_record"].astype(float)
        + 0.12 * out["has_literature_support"].astype(float)
        + 0.10 * out["is_mutant_assay"].astype(float)
        + 0.08 * out["is_cellular_assay"].astype(float)
        + 0.06 * out["is_panel_assay"].astype(float)
        - 0.18 * out["is_virtual_proxy_assay"].astype(float)
    ).clip(lower=0.0, upper=1.0)
    return out


def _aggregate_assay_catalog(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    catalog = (
        df.groupby("AID", as_index=False)
        .agg(
            assay_name=("Assay Name", "first"),
            assay_type=("assay_type", "first"),
            target_accession=("Target Accession", "first"),
            n_records=("AID", "size"),
            n_unique_cids=("CID", lambda s: int(pd.Series(s).nunique())),
            active_records=("is_active_record", "sum"),
            confirmatory_records=("is_confirmatory_record", "sum"),
            direct_kinase_records=("is_direct_kinase_assay", "sum"),
            target_specific_records=("is_target_specific_assay", "sum"),
            mutant_records=("is_mutant_assay", "sum"),
            cellular_records=("is_cellular_assay", "sum"),
            literature_records=("has_literature_support", "sum"),
            virtual_proxy_records=("is_virtual_proxy_assay", "sum"),
            median_activity_uM=("Activity Value [uM]", "median"),
            best_proxy_pIC50=("activity_proxy_pIC50", "max"),
            mean_record_relevance=("record_relevance_score", "mean"),
        )
        .reset_index(drop=True)
    )
    catalog["activity_fraction"] = catalog["active_records"] / catalog["n_records"].clip(lower=1)
    catalog["virtual_proxy_fraction"] = catalog["virtual_proxy_records"] / catalog["n_records"].clip(lower=1)
    catalog["assay_support_tier"] = pd.cut(
        catalog["mean_record_relevance"].fillna(0.0),
        bins=[-0.001, 0.25, 0.55, 1.0],
        labels=["weak", "moderate", "strong"],
    ).astype(str)
    return catalog.sort_values(
        ["mean_record_relevance", "activity_fraction", "best_proxy_pIC50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_processed_reference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = df[pd.to_numeric(df["Target GeneID"], errors="coerce") == HUMAN_EGFR_GENE_ID].copy()
    filtered["CID"] = pd.to_numeric(filtered["CID"], errors="coerce")
    filtered = filtered[filtered["CID"].notna()].copy()
    if filtered.empty:
        return filtered, filtered, pd.DataFrame()

    filtered["CID"] = filtered["CID"].astype(int)
    filtered["Activity Value [uM]"] = pd.to_numeric(filtered["Activity Value [uM]"], errors="coerce")
    filtered["activity_proxy_pIC50"] = filtered["Activity Value [uM]"].map(_activity_um_to_pic50)

    properties = _fetch_compound_properties(sorted(filtered["CID"].dropna().unique().tolist()))
    properties["CID"] = pd.to_numeric(properties["CID"], errors="coerce").astype("Int64")
    properties["smiles"] = properties["ConnectivitySMILES"].map(canonicalize_smiles)
    properties = properties.rename(columns={"IUPACName": "iupac_name"})

    merged = filtered.merge(properties[["CID", "ConnectivitySMILES", "iupac_name", "smiles"]], on="CID", how="left")
    merged = merged[merged["smiles"].notna()].copy()
    merged["activity_outcome"] = merged["Activity Outcome"].fillna("Unspecified").astype(str)
    merged["assay_type"] = merged["Assay Type"].fillna("Unknown").astype(str)
    merged["pubmed_id"] = pd.to_numeric(merged["PubMed ID"], errors="coerce")
    merged = _flag_assay_metadata(merged)
    assay_catalog = _aggregate_assay_catalog(merged)

    grouped = (
        merged.groupby("CID", as_index=False)
        .agg(
            smiles=("smiles", "first"),
            iupac_name=("iupac_name", "first"),
            n_records=("CID", "size"),
            unique_assay_count=("AID", lambda s: int(pd.Series(s).nunique())),
            active_records=("activity_outcome", lambda s: int((s == "Active").sum())),
            inactive_records=("activity_outcome", lambda s: int((s == "Inactive").sum())),
            unspecified_records=("activity_outcome", lambda s: int((s == "Unspecified").sum())),
            confirmatory_records=("assay_type", lambda s: int((s == "Confirmatory").sum())),
            screening_records=("assay_type", lambda s: int((s == "Screening").sum())),
            min_activity_uM=("Activity Value [uM]", "min"),
            median_activity_uM=("Activity Value [uM]", "median"),
            best_proxy_pIC50=("activity_proxy_pIC50", "max"),
            median_proxy_pIC50=("activity_proxy_pIC50", "median"),
            assay_name=("Assay Name", "first"),
            assay_type=("assay_type", "first"),
            pubmed_count=("pubmed_id", lambda s: int(pd.Series(s).dropna().nunique())),
            direct_kinase_records=("is_direct_kinase_assay", "sum"),
            target_specific_records=("is_target_specific_assay", "sum"),
            cellular_records=("is_cellular_assay", "sum"),
            mutant_records=("is_mutant_assay", "sum"),
            literature_records=("has_literature_support", "sum"),
            virtual_proxy_records=("is_virtual_proxy_assay", "sum"),
            panel_records=("is_panel_assay", "sum"),
            active_confirmatory_records=("is_confirmatory_record", lambda s: int((pd.Series(s).astype(bool)).sum())),
            active_direct_kinase_records=("is_direct_kinase_assay", lambda s: int((pd.Series(s).astype(bool)).sum())),
            mean_record_relevance=("record_relevance_score", "mean"),
        )
        .reset_index(drop=True)
    )

    total_outcome_records = (grouped["active_records"] + grouped["inactive_records"]).clip(lower=1)
    grouped["active_fraction"] = grouped["active_records"] / total_outcome_records
    grouped["confirmatory_fraction"] = grouped["confirmatory_records"] / grouped["n_records"].clip(lower=1)
    grouped["direct_kinase_fraction"] = grouped["direct_kinase_records"] / grouped["n_records"].clip(lower=1)
    grouped["target_specific_fraction"] = grouped["target_specific_records"] / grouped["n_records"].clip(lower=1)
    grouped["cellular_fraction"] = grouped["cellular_records"] / grouped["n_records"].clip(lower=1)
    grouped["mutant_fraction"] = grouped["mutant_records"] / grouped["n_records"].clip(lower=1)
    grouped["literature_fraction"] = grouped["literature_records"] / grouped["n_records"].clip(lower=1)
    grouped["panel_fraction"] = grouped["panel_records"] / grouped["n_records"].clip(lower=1)
    grouped["virtual_proxy_fraction"] = grouped["virtual_proxy_records"] / grouped["n_records"].clip(lower=1)
    grouped["assay_count_support"] = grouped["n_records"].map(lambda n: min(1.0, math.log1p(float(n)) / math.log(8.0)))
    grouped["assay_diversity_score"] = grouped["unique_assay_count"].map(lambda n: min(1.0, math.log1p(float(n)) / math.log(10.0)))
    grouped["potency_support"] = grouped["best_proxy_pIC50"].fillna(0.0).map(lambda value: max(0.0, min(1.0, (float(value) - 5.5) / 3.5)))
    grouped["pubchem_evidence_score"] = (
        0.42 * grouped["active_fraction"].fillna(0.0)
        + 0.18 * grouped["confirmatory_fraction"].fillna(0.0)
        + 0.20 * grouped["assay_count_support"].fillna(0.0)
        + 0.20 * grouped["potency_support"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    grouped["pubchem_relevance_score"] = (
        0.24 * grouped["target_specific_fraction"].fillna(0.0)
        + 0.18 * grouped["direct_kinase_fraction"].fillna(0.0)
        + 0.12 * grouped["confirmatory_fraction"].fillna(0.0)
        + 0.10 * grouped["literature_fraction"].fillna(0.0)
        + 0.10 * grouped["mutant_fraction"].fillna(0.0)
        + 0.08 * grouped["cellular_fraction"].fillna(0.0)
        + 0.08 * grouped["assay_diversity_score"].fillna(0.0)
        + 0.05 * grouped["panel_fraction"].fillna(0.0)
        - 0.15 * grouped["virtual_proxy_fraction"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    grouped["pubchem_orthogonal_support_score"] = (
        0.35 * grouped["cellular_fraction"].fillna(0.0)
        + 0.25 * grouped["mutant_fraction"].fillna(0.0)
        + 0.20 * grouped["literature_fraction"].fillna(0.0)
        + 0.20 * grouped["assay_diversity_score"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    grouped["pubchem_enriched_evidence_score"] = (
        0.24 * grouped["active_fraction"].fillna(0.0)
        + 0.14 * grouped["confirmatory_fraction"].fillna(0.0)
        + 0.12 * grouped["assay_count_support"].fillna(0.0)
        + 0.12 * grouped["potency_support"].fillna(0.0)
        + 0.14 * grouped["pubchem_relevance_score"].fillna(0.0)
        + 0.10 * grouped["pubchem_orthogonal_support_score"].fillna(0.0)
        + 0.08 * grouped["assay_diversity_score"].fillna(0.0)
        + 0.06 * (1.0 - grouped["virtual_proxy_fraction"].fillna(0.0))
    ).clip(lower=0.0, upper=1.0)
    grouped["consensus_active"] = (
        (grouped["active_records"] >= 1)
        & (
            (grouped["active_fraction"] >= 0.50)
            | (grouped["best_proxy_pIC50"].fillna(0.0) >= 6.0)
        )
        & (grouped["pubchem_relevance_score"].fillna(0.0) >= 0.22)
        & (grouped["virtual_proxy_fraction"].fillna(0.0) <= 0.80)
    )
    grouped["evidence_tier"] = pd.cut(
        grouped["pubchem_enriched_evidence_score"],
        bins=[-0.001, 0.35, 0.60, 1.0],
        labels=["weak", "moderate", "strong"],
    ).astype(str)
    grouped["pubchem_signal_profile"] = grouped.apply(
        lambda row: ";".join(
            bit
            for bit, flag in [
                ("direct_kinase", row.get("direct_kinase_fraction", 0.0) >= 0.30),
                ("target_specific", row.get("target_specific_fraction", 0.0) >= 0.30),
                ("mutant", row.get("mutant_fraction", 0.0) >= 0.20),
                ("cellular", row.get("cellular_fraction", 0.0) >= 0.20),
                ("literature", row.get("literature_fraction", 0.0) >= 0.10),
                ("panel", row.get("panel_fraction", 0.0) >= 0.30),
                ("virtual_proxy_exposed", row.get("virtual_proxy_fraction", 0.0) >= 0.25),
            ]
            if flag
        ) or None,
        axis=1,
    )
    grouped["source_dataset"] = "pubchem_bioassay"
    grouped = grouped.sort_values(
        ["consensus_active", "pubchem_enriched_evidence_score", "best_proxy_pIC50", "n_records"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return merged.reset_index(drop=True), grouped, assay_catalog


def ensure_pubchem_reference() -> Path:
    processed_path = PROCESSED_DIR / "pubchem_egfr_reference.csv"
    summary_path = PROCESSED_DIR / "pubchem_egfr_reference.summary.json"
    if processed_path.exists():
        try:
            existing = pd.read_csv(processed_path, nrows=3, low_memory=False)
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            if REQUIRED_PROCESSED_COLUMNS.issubset(existing.columns) and summary.get("schema_version") == PUBCHEM_SCHEMA_VERSION:
                return processed_path
        except Exception:
            pass
    main()
    return processed_path


def main() -> None:
    PUBCHEM_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    concise_df = _fetch_csv(PUBCHEM_EGFR_CONCISE_URL)
    interim, processed, assay_catalog = _build_processed_reference(concise_df)

    interim_path = INTERIM_DIR / "pubchem_egfr_bioassay_interim.csv"
    processed_path = PROCESSED_DIR / "pubchem_egfr_reference.csv"
    assay_catalog_path = PROCESSED_DIR / "pubchem_egfr_assay_catalog.csv"
    summary_path = PROCESSED_DIR / "pubchem_egfr_reference.summary.json"

    interim.to_csv(interim_path, index=False)
    processed.to_csv(processed_path, index=False)
    assay_catalog.to_csv(assay_catalog_path, index=False)

    summary = {
        "schema_version": PUBCHEM_SCHEMA_VERSION,
        "n_rows_target_filtered": int(len(interim)),
        "n_unique_cids": int(processed["CID"].nunique()) if not processed.empty else 0,
        "n_consensus_active": int(processed["consensus_active"].fillna(False).sum()) if not processed.empty else 0,
        "mean_evidence_score": float(processed["pubchem_evidence_score"].mean()) if not processed.empty else None,
        "mean_enriched_evidence_score": float(processed["pubchem_enriched_evidence_score"].mean()) if not processed.empty else None,
        "mean_relevance_score": float(processed["pubchem_relevance_score"].mean()) if not processed.empty else None,
        "mean_orthogonal_support_score": float(processed["pubchem_orthogonal_support_score"].mean()) if not processed.empty else None,
        "strong_evidence_rate": float((processed["evidence_tier"] == "strong").mean()) if not processed.empty else None,
        "virtual_proxy_exposed_rate": float((processed["virtual_proxy_fraction"].fillna(0.0) >= 0.25).mean()) if not processed.empty else None,
        "n_assays": int(assay_catalog["AID"].nunique()) if not assay_catalog.empty else 0,
        "median_best_proxy_pIC50": float(processed["best_proxy_pIC50"].median()) if "best_proxy_pIC50" in processed.columns and not processed["best_proxy_pIC50"].dropna().empty else None,
        "source_url": PUBCHEM_EGFR_CONCISE_URL,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved PubChem EGFR interim interactions: {interim_path}")
    print(f"[OK] Saved PubChem EGFR reference: {processed_path}")
    print(f"[OK] Saved PubChem EGFR assay catalog: {assay_catalog_path}")
    print(f"[OK] Saved PubChem EGFR summary: {summary_path}")
    preview_cols = [
        "CID",
        "smiles",
        "active_records",
        "inactive_records",
        "best_proxy_pIC50",
        "pubchem_enriched_evidence_score",
        "pubchem_relevance_score",
        "virtual_proxy_fraction",
        "consensus_active",
    ]
    preview_cols = [column for column in preview_cols if column in processed.columns]
    print(processed[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
