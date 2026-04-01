from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

from src.config import (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    KEEP_STANDARD_TYPE, KEEP_STANDARD_UNITS, KEEP_STANDARD_RELATION,
    IC50_NM_MIN, IC50_NM_MAX
)
from src.data.chembl_document_years import fetch_document_year_map
from src.utils.chem import canonicalize_smiles, ic50_nm_to_pic50


def fetch_smiles_map(mol_ids: list[str], batch_size: int = 200, sleep_s: float = 0.1) -> dict[str, str | None]:
    """
    Fetch canonical SMILES for ChEMBL molecule IDs in batches.
    """
    molecule = new_client.molecule
    out: dict[str, str | None] = {}

    for i in tqdm(range(0, len(mol_ids), batch_size), desc="Fetching SMILES"):
        batch = mol_ids[i:i + batch_size]
        mols = molecule.filter(molecule_chembl_id__in=batch).only(["molecule_chembl_id", "molecule_structures"])

        for m in mols:
            mid = m.get("molecule_chembl_id")
            ms = m.get("molecule_structures") or {}
            out[mid] = ms.get("canonical_smiles")

        time.sleep(sleep_s)

    return out


def _join_unique_non_missing(values: pd.Series) -> str | pd.NA:
    items: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "missing":
            continue
        items.append(text)
    unique_items = sorted(set(items))
    return ";".join(unique_items) if unique_items else pd.NA


def _count_unique_non_missing(values: pd.Series) -> int:
    unique_items: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "missing":
            continue
        unique_items.add(text)
    return int(len(unique_items))


def clean_raw_to_processed(raw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - interim_df: measurement-level cleaned data
      - processed_df: molecule-level deduped data (median aggregation)
    """
    df = pd.read_csv(raw_csv, low_memory=False)

    # Basic schema sanity
    required_cols = {"molecule_chembl_id", "standard_type", "standard_units", "standard_relation", "standard_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}. Found columns: {list(df.columns)}")

    # Filters
    df = df[df["standard_type"].isin(KEEP_STANDARD_TYPE)]
    df = df[df["standard_units"].isin(KEEP_STANDARD_UNITS)]
    df = df[df["standard_relation"].isin(KEEP_STANDARD_RELATION)]
    df = df[df["standard_value"].notna()].copy()

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"].notna()].copy()

    # IC50 range filter (nM)
    df = df[(df["standard_value"] >= IC50_NM_MIN) & (df["standard_value"] <= IC50_NM_MAX)].copy()

    # Fetch SMILES
    mol_ids = sorted(df["molecule_chembl_id"].dropna().unique().tolist())
    smiles_map = fetch_smiles_map(mol_ids)

    df["smiles_raw"] = df["molecule_chembl_id"].map(smiles_map)
    df["smiles_canonical"] = df["smiles_raw"].apply(canonicalize_smiles)
    df = df[df["smiles_canonical"].notna()].copy()

    # Compute pIC50
    df["ic50_nm"] = df["standard_value"].astype(float)
    df["pIC50"] = df["ic50_nm"].apply(ic50_nm_to_pic50)
    df = df[df["pIC50"].notna()].copy()

    if "year" not in df.columns:
        df["year"] = np.nan
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_source"] = pd.NA
    df["year_confidence"] = np.nan

    raw_year_mask = df["year"].notna()
    df.loc[raw_year_mask, "year_source"] = "raw_year"
    df.loc[raw_year_mask, "year_confidence"] = 1.0

    if "document_chembl_id" in df.columns:
        document_ids = df["document_chembl_id"].dropna().astype(str).tolist()
        doc_year_map = fetch_document_year_map(document_ids) if document_ids else {}
        if doc_year_map:
            doc_years = pd.to_numeric(df["document_chembl_id"].map(doc_year_map), errors="coerce")
            filled_mask = df["year"].isna() & doc_years.notna()
            df.loc[filled_mask, "year"] = doc_years[filled_mask]
            df.loc[filled_mask, "year_source"] = "document_chembl_id"
            df.loc[filled_mask, "year_confidence"] = 0.95

    df.loc[df["year"].notna() & df["year_source"].isna(), "year_source"] = "raw_year"
    df.loc[df["year"].notna() & df["year_confidence"].isna(), "year_confidence"] = 1.0
    df.loc[df["year"].isna() & df["year_source"].isna(), "year_source"] = "missing"
    df["year_confidence"] = pd.to_numeric(df["year_confidence"], errors="coerce").fillna(0.0)

    interim_df = df.reset_index(drop=True)

    aggregation = {
        "ic50_nm_median": ("ic50_nm", "median"),
        "pIC50_median": ("pIC50", "median"),
        "n_measurements": ("pIC50", "size"),
        "temporal_year_min": ("year", "min"),
        "temporal_year_max": ("year", "max"),
        "temporal_year_coverage_rate": ("year", lambda values: float(values.notna().mean())),
        "temporal_year_source_count": ("year_source", _count_unique_non_missing),
        "temporal_year_sources": ("year_source", _join_unique_non_missing),
        "temporal_year_confidence_mean": ("year_confidence", "mean"),
    }

    if "year" in interim_df.columns and interim_df["year"].notna().any():
        aggregation["year_min"] = ("year", "min")
        aggregation["year_max"] = ("year", "max")

    processed_df = (
        interim_df.groupby("smiles_canonical", as_index=False)
        .agg(**aggregation)
        .sort_values("pIC50_median", ascending=False)
        .reset_index(drop=True)
    )

    for column in ["year_min", "year_max", "temporal_year_min", "temporal_year_max"]:
        if column not in processed_df.columns:
            processed_df[column] = np.nan
    if "temporal_year_coverage_rate" not in processed_df.columns:
        processed_df["temporal_year_coverage_rate"] = np.nan
    if "temporal_year_source_count" not in processed_df.columns:
        processed_df["temporal_year_source_count"] = 0
    if "temporal_year_sources" not in processed_df.columns:
        processed_df["temporal_year_sources"] = pd.NA
    if "temporal_year_confidence_mean" not in processed_df.columns:
        processed_df["temporal_year_confidence_mean"] = np.nan

    return interim_df, processed_df


def main() -> None:
    print("CLEAN SCRIPT STARTED", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_DIR / "chembl_egfr_ic50_raw.csv"
    print(f"[INFO] raw_csv: {raw_csv}", flush=True)
    print(f"[INFO] exists: {raw_csv.exists()}", flush=True)

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Missing raw file: {raw_csv}\n"
            "Run: python -m src.data.fetch_chembl_egfr"
        )

    interim_df, processed_df = clean_raw_to_processed(raw_csv)

    interim_path = INTERIM_DIR / "chembl_egfr_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "egfr_chembl_ic50_clean.csv"

    interim_df.to_csv(interim_path, index=False)
    processed_df.to_csv(processed_path, index=False)

    print(f"[OK] Interim saved:    {interim_path}  (rows={len(interim_df)})", flush=True)
    print(f"[OK] Processed saved: {processed_path} (molecules={len(processed_df)})", flush=True)
    print(processed_df.head(5), flush=True)


if __name__ == "__main__":
    main()
