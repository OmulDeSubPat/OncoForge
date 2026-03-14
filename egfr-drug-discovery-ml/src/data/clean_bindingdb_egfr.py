from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.config import IC50_NM_MAX, IC50_NM_MIN, INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from src.data.fetch_bindingdb_egfr import download_bindingdb_zip
from src.utils.chem import canonicalize_smiles, ic50_nm_to_pic50


EGFR_UNIPROT = "P00533"
EGFR_NAME_PATTERNS = (
    "epidermal growth factor receptor",
    "egfr",
    "erbb1",
    "receptor tyrosine-protein kinase erb-b1",
)


def _default_bindingdb_archive(variant: str) -> Path:
    pattern = re.compile(rf"bindingdb_{re.escape(variant)}_(\d{{6}})_tsv\.zip$")
    candidates: list[tuple[str, Path]] = []
    for path in RAW_DIR.glob(f"bindingdb_{variant}_*_tsv.zip"):
        match = pattern.match(path.name)
        if match:
            candidates.append((match.group(1), path))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]
    return RAW_DIR / f"bindingdb_{variant}_latest_tsv.zip"


def _match_target(chunk: pd.DataFrame) -> pd.Series:
    target_name = chunk.get("Target Name", pd.Series("", index=chunk.index)).fillna("").astype(str).str.lower()
    recommended = chunk.get(
        "UniProt (SwissProt) Recommended Name of Target Chain 1",
        pd.Series("", index=chunk.index),
    ).fillna("").astype(str).str.lower()
    entry_name = chunk.get(
        "UniProt (SwissProt) Entry Name of Target Chain 1",
        pd.Series("", index=chunk.index),
    ).fillna("").astype(str).str.lower()
    primary_id = chunk.get(
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        pd.Series("", index=chunk.index),
    ).fillna("").astype(str).str.upper()
    organism = chunk.get(
        "Target Source Organism According to Curator or DataSource",
        pd.Series("", index=chunk.index),
    ).fillna("").astype(str).str.lower()

    name_mask = pd.Series(False, index=chunk.index)
    for pattern in EGFR_NAME_PATTERNS:
        name_mask = name_mask | target_name.str.contains(pattern, regex=False) | recommended.str.contains(pattern, regex=False) | entry_name.str.contains(pattern, regex=False)

    organism_mask = organism.eq("homo sapiens") | organism.eq("")
    uniprot_mask = primary_id.eq(EGFR_UNIPROT)
    return (name_mask | uniprot_mask) & organism_mask


def _open_bindingdb_member(zip_path: Path):
    archive = ZipFile(zip_path)
    members = [member for member in archive.namelist() if member.lower().endswith(".tsv")]
    if not members:
        archive.close()
        raise FileNotFoundError(f"No TSV member found inside {zip_path}")
    return archive, members[0]


def clean_bindingdb_archive(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    archive, member_name = _open_bindingdb_member(zip_path)

    rows: list[pd.DataFrame] = []
    total_rows = 0
    matched_rows = 0

    try:
        with archive.open(member_name) as handle:
            reader = pd.read_csv(
                handle,
                sep="\t",
                chunksize=50000,
                dtype=str,
                low_memory=False,
            )
            for chunk_idx, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)
                chunk = chunk[_match_target(chunk)].copy()
                matched_rows += len(chunk)
                if chunk.empty:
                    continue
                chunk["IC50 (nM)"] = pd.to_numeric(chunk["IC50 (nM)"], errors="coerce")
                chunk = chunk[chunk["IC50 (nM)"].notna()].copy()
                chunk = chunk[
                    (chunk["IC50 (nM)"] >= IC50_NM_MIN)
                    & (chunk["IC50 (nM)"] <= IC50_NM_MAX)
                ].copy()
                if chunk.empty:
                    continue

                chunk["smiles_raw"] = chunk["Ligand SMILES"].astype(str)
                chunk["smiles_canonical"] = chunk["smiles_raw"].apply(canonicalize_smiles)
                chunk = chunk[chunk["smiles_canonical"].notna()].copy()
                if chunk.empty:
                    continue

                chunk["ic50_nm"] = chunk["IC50 (nM)"].astype(float)
                chunk["pIC50"] = chunk["ic50_nm"].apply(ic50_nm_to_pic50)
                chunk = chunk[chunk["pIC50"].notna()].copy()
                if chunk.empty:
                    continue

                publication_year = pd.to_datetime(
                    chunk.get("Date of publication"),
                    errors="coerce",
                ).dt.year

                curated = pd.DataFrame(
                    {
                        "source_dataset": "bindingdb_articles",
                        "source_record_id": chunk.get("BindingDB Reactant_set_id"),
                        "bindingdb_reactant_set_id": chunk.get("BindingDB Reactant_set_id"),
                        "bindingdb_monomer_id": chunk.get("BindingDB MonomerID"),
                        "target_name": chunk.get("Target Name"),
                        "target_organism": chunk.get("Target Source Organism According to Curator or DataSource"),
                        "uniprot_primary_id": chunk.get("UniProt (SwissProt) Primary ID of Target Chain 1"),
                        "smiles_raw": chunk["smiles_raw"],
                        "smiles_canonical": chunk["smiles_canonical"],
                        "ic50_nm": chunk["ic50_nm"],
                        "pIC50": chunk["pIC50"],
                        "publication_date": chunk.get("Date of publication"),
                        "bindingdb_date": chunk.get("Date in BindingDB"),
                        "year": publication_year,
                        "curation_source": chunk.get("Curation/DataSource"),
                        "article_doi": chunk.get("Article DOI"),
                        "pmid": chunk.get("PMID"),
                        "chembl_id": chunk.get("ChEMBL ID of Ligand"),
                    }
                )
                rows.append(curated)
                print(
                    f"[INFO] BindingDB chunk {chunk_idx}: kept {len(curated)} EGFR IC50 rows",
                    flush=True,
                )
    finally:
        archive.close()

    if not rows:
        raise ValueError("No EGFR IC50 rows were extracted from the BindingDB archive.")

    interim_df = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["source_record_id", "smiles_canonical", "ic50_nm"],
    ).reset_index(drop=True)

    processed_df = (
        interim_df.groupby("smiles_canonical", as_index=False)
        .agg(
            ic50_nm_median=("ic50_nm", "median"),
            pIC50_median=("pIC50", "median"),
            n_measurements=("pIC50", "size"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            n_sources=("source_dataset", "nunique"),
        )
        .sort_values("pIC50_median", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "archive": str(zip_path),
        "total_rows_scanned": int(total_rows),
        "target_matched_rows": int(matched_rows),
        "interim_rows": int(len(interim_df)),
        "unique_molecules": int(len(processed_df)),
    }
    return interim_df, processed_df, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean a BindingDB archive into EGFR IC50 artifacts.")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Local BindingDB TSV zip path. If omitted, use the latest downloaded archive.",
    )
    parser.add_argument(
        "--variant",
        choices=["articles", "all"],
        default="articles",
        help="Which BindingDB variant to use when auto-downloading.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the BindingDB archive if it is not already present.",
    )
    args = parser.parse_args(argv)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip_path) if args.zip_path else _default_bindingdb_archive(args.variant)
    if not zip_path.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"Missing BindingDB archive: {zip_path}\n"
                "Run: python -m src.data.fetch_bindingdb_egfr --variant "
                f"{args.variant}"
            )
        zip_path = download_bindingdb_zip(variant=args.variant)

    interim_df, processed_df, summary = clean_bindingdb_archive(zip_path)

    interim_path = INTERIM_DIR / "bindingdb_egfr_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "egfr_bindingdb_ic50_clean.csv"
    summary_path = PROCESSED_DIR / "egfr_bindingdb_summary.json"

    interim_df.to_csv(interim_path, index=False)
    processed_df.to_csv(processed_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Interim saved:    {interim_path}  (rows={len(interim_df)})")
    print(f"[OK] Processed saved: {processed_path} (molecules={len(processed_df)})")
    print(f"[OK] Summary saved:   {summary_path}")
    print(processed_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
