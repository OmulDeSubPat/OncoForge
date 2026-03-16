from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from src.utils.chem import canonicalize_smiles, ic50_nm_to_pic50


IUPHAR_DIR = EXTERNAL_DIR / "iuphar"
LIGANDS_URL = "https://www.guidetopharmacology.org/DATA/ligands.csv"
CATALYTIC_RECEPTOR_INTERACTIONS_URL = "https://www.guidetopharmacology.org/DATA/catalytic_receptor_interactions.csv"
EGFR_TARGET_GENE_SYMBOL = "EGFR"


def _load_remote_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, skiprows=1, low_memory=False)


def _coerce_pic50(row: pd.Series) -> float | None:
    affinity_units = str(row.get("Affinity Units", "") or "").strip()
    affinity_median = pd.to_numeric(row.get("Affinity Median"), errors="coerce")
    if affinity_units == "pIC50" and pd.notna(affinity_median):
        return float(affinity_median)

    original_ic50_nm = pd.to_numeric(row.get("Original Affinity Median nm"), errors="coerce")
    if pd.notna(original_ic50_nm):
        return ic50_nm_to_pic50(float(original_ic50_nm))

    return None


def _build_processed_reference(interactions: pd.DataFrame, ligands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = interactions[
        (interactions["Target Gene Symbol"].fillna("").astype(str).str.upper() == EGFR_TARGET_GENE_SYMBOL)
        & (interactions["Target Species"].fillna("").astype(str) == "Human")
    ].copy()
    if filtered.empty:
        return filtered, filtered

    ligand_cols = [
        "Ligand ID",
        "Name",
        "SMILES",
        "InChIKey",
        "PubChem CID",
        "ChEMBL ID",
    ]
    ligand_frame = ligands[[column for column in ligand_cols if column in ligands.columns]].copy()
    ligand_frame = ligand_frame.rename(
        columns={
            "Name": "Ligand Name Catalog",
            "SMILES": "Ligand SMILES",
            "InChIKey": "Ligand InChIKey",
            "PubChem CID": "Ligand PubChem CID",
            "ChEMBL ID": "Ligand ChEMBL ID",
        }
    )

    out = filtered.merge(ligand_frame, on="Ligand ID", how="left")
    out["smiles"] = out["Ligand SMILES"].map(canonicalize_smiles)
    out["pIC50"] = out.apply(_coerce_pic50, axis=1)
    out["pubmed_id"] = pd.to_numeric(out.get("PubMed ID"), errors="coerce")
    out = out[out["smiles"].notna() & out["pIC50"].notna()].copy()
    if out.empty:
        return out, out

    out["source_dataset"] = "iuphar"
    out["ligand_name"] = out["Ligand"].fillna(out["Ligand Name Catalog"]).astype(str)
    out["year"] = pd.to_numeric(out["pubmed_id"], errors="coerce")
    keep_cols = [
        "source_dataset",
        "Ligand ID",
        "ligand_name",
        "smiles",
        "pIC50",
        "Ligand ChEMBL ID",
        "Ligand PubChem CID",
        "Ligand InChIKey",
        "PubMed ID",
        "Affinity Units",
        "Original Affinity Median nm",
        "Action",
        "Type",
        "Primary Target",
    ]
    interim = out[[column for column in keep_cols if column in out.columns]].copy()
    interim = interim.rename(
        columns={
            "Ligand ID": "iuphar_ligand_id",
            "Ligand ChEMBL ID": "chembl_id",
            "Ligand PubChem CID": "pubchem_cid",
            "Ligand InChIKey": "inchikey",
            "PubMed ID": "pubmed_id",
            "Original Affinity Median nm": "original_affinity_median_nm",
            "Primary Target": "primary_target",
        }
    )

    processed = (
        interim.groupby("smiles", as_index=False)
        .agg(
            pIC50_median=("pIC50", "median"),
            n_iuphar_records=("pIC50", "size"),
            ligand_name=("ligand_name", "first"),
            chembl_id=("chembl_id", "first"),
            pubchem_cid=("pubchem_cid", "first"),
            inchikey=("inchikey", "first"),
            pubmed_count=("pubmed_id", lambda values: int(pd.Series(values).dropna().nunique())),
            action=("Action", "first"),
            ligand_type=("Type", "first"),
        )
        .sort_values(["pIC50_median", "n_iuphar_records"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return interim.reset_index(drop=True), processed


def ensure_iuphar_reference() -> Path:
    processed_path = PROCESSED_DIR / "iuphar_egfr_reference.csv"
    if processed_path.exists():
        return processed_path
    main()
    return processed_path


def main() -> None:
    IUPHAR_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ligands = _load_remote_csv(LIGANDS_URL)
    interactions = _load_remote_csv(CATALYTIC_RECEPTOR_INTERACTIONS_URL)
    interim, processed = _build_processed_reference(interactions, ligands)

    interim_path = INTERIM_DIR / "iuphar_egfr_interactions_interim.csv"
    processed_path = PROCESSED_DIR / "iuphar_egfr_reference.csv"
    summary_path = PROCESSED_DIR / "iuphar_egfr_reference.summary.json"

    interim.to_csv(interim_path, index=False)
    processed.to_csv(processed_path, index=False)
    summary = {
        "n_interactions": int(len(interim)),
        "n_unique_molecules": int(len(processed)),
        "max_pic50": float(processed["pIC50_median"].max()) if not processed.empty else None,
        "median_pic50": float(processed["pIC50_median"].median()) if not processed.empty else None,
        "n_with_chembl_id": int(processed["chembl_id"].fillna("").astype(str).str.len().gt(0).sum()) if "chembl_id" in processed.columns else 0,
        "source_url_ligands": LIGANDS_URL,
        "source_url_interactions": CATALYTIC_RECEPTOR_INTERACTIONS_URL,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved IUPHAR EGFR interim interactions: {interim_path}")
    print(f"[OK] Saved IUPHAR EGFR reference: {processed_path}")
    print(f"[OK] Saved IUPHAR EGFR summary: {summary_path}")
    print(processed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
