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


def clean_raw_to_processed(raw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - interim_df: measurement-level cleaned data
      - processed_df: molecule-level deduped data (median aggregation)
    """
    df = pd.read_csv(raw_csv)

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

    interim_df = df.reset_index(drop=True)

    processed_df = interim_df.groupby("smiles_canonical", as_index=False).agg(
        ic50_nm_median=("ic50_nm", "median"),
        pIC50_median=("pIC50", "median"),
        n_measurements=("pIC50", "size"),
        year_min=("year", "min") if "year" in interim_df.columns else ("pIC50", "min"),
        year_max=("year", "max") if "year" in interim_df.columns else ("pIC50", "max"),
    ).sort_values("pIC50_median", ascending=False).reset_index(drop=True)

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