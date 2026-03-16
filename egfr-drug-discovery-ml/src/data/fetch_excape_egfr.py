from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import requests

from src.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from src.utils.chem import canonicalize_smiles


EXCAPE_DIR = EXTERNAL_DIR / "excape"
EXCAPE_FILENAME = "pubchem.chembl.dataset4publication_inchi_smiles.tsv.xz"
EXCAPE_URL = f"https://zenodo.org/records/173258/files/{EXCAPE_FILENAME}?download=1"
EXCAPE_CHUNK_SIZE = 200_000


def _download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def _pxc50_to_ic50_nm(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return float(10 ** (9.0 - numeric))


def _iter_target_chunks(dataset_path: Path):
    chunk_iter = pd.read_csv(
        dataset_path,
        sep="\t",
        compression="xz",
        chunksize=EXCAPE_CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in chunk_iter:
        gene = chunk.get("Gene_Symbol", pd.Series("", index=chunk.index)).astype(str)
        tax_id = pd.to_numeric(chunk.get("Tax_ID"), errors="coerce")
        mask = gene.str.upper().eq("EGFR") & tax_id.eq(9606)
        if mask.any():
            yield chunk.loc[mask].copy()


def _coerce_excape(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out["smiles_canonical"] = out["SMILES"].map(canonicalize_smiles)
    out["pIC50"] = pd.to_numeric(out.get("pXC50"), errors="coerce")
    out["ic50_nm"] = out["pIC50"].map(_pxc50_to_ic50_nm)
    out["year"] = pd.NA
    out["source_dataset"] = (
        "excape_" + out.get("DB", pd.Series("unknown", index=out.index)).astype(str).str.lower()
    )
    out["source_record_id"] = out.get("Original_Entry_ID", pd.Series(range(1, len(out) + 1), index=out.index)).astype(str)
    out["excape_activity_flag"] = out.get("Activity_Flag", pd.Series("", index=out.index)).astype(str)
    out["excape_assay_id"] = out.get("Original_Assay_ID", pd.Series("", index=out.index)).astype(str)
    out = out[
        out["smiles_canonical"].notna()
        & out["pIC50"].notna()
        & out["ic50_nm"].notna()
        & out["pIC50"].between(4.0, 11.5, inclusive="both")
    ].copy()

    interim = out[
        [
            "source_dataset",
            "source_record_id",
            "smiles_canonical",
            "ic50_nm",
            "pIC50",
            "year",
            "excape_activity_flag",
            "excape_assay_id",
            "Gene_Symbol",
            "DB",
        ]
    ].reset_index(drop=True)
    processed = (
        interim.groupby("smiles_canonical", as_index=False)
        .agg(
            pIC50_median=("pIC50", "median"),
            ic50_nm_median=("ic50_nm", "median"),
            n_measurements=("pIC50", "size"),
            active_fraction=("excape_activity_flag", lambda values: float((pd.Series(values).astype(str) == "A").mean())),
            source_dbs=("DB", lambda values: ";".join(sorted(set(str(value) for value in values if value)))),
            assay_count=("excape_assay_id", lambda values: int(pd.Series(values).replace("", pd.NA).dropna().nunique())),
        )
        .sort_values(["pIC50_median", "n_measurements"], ascending=[False, False])
        .reset_index(drop=True)
    )
    processed["source_dataset"] = "excape"
    processed["excape_support_score"] = (
        0.40 * processed["pIC50_median"].map(lambda value: max(0.0, min(1.0, (float(value) - 5.5) / 3.5)))
        + 0.25 * processed["active_fraction"].fillna(0.0)
        + 0.20 * processed["n_measurements"].map(lambda value: min(1.0, math.log1p(float(value)) / math.log(12.0)))
        + 0.15 * processed["assay_count"].map(lambda value: min(1.0, math.log1p(float(value)) / math.log(10.0)))
    ).clip(lower=0.0, upper=1.0)
    return interim, processed


def ensure_excape_reference() -> Path:
    processed_path = PROCESSED_DIR / "excape_egfr_reference.csv"
    if processed_path.exists():
        return processed_path
    main()
    return processed_path


def main() -> None:
    EXCAPE_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = _download_if_missing(EXCAPE_URL, EXCAPE_DIR / EXCAPE_FILENAME)
    frames = list(_iter_target_chunks(dataset_path))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    interim, processed = _coerce_excape(raw)

    interim_path = INTERIM_DIR / "excape_egfr_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "excape_egfr_reference.csv"
    summary_path = PROCESSED_DIR / "excape_egfr_reference.summary.json"

    interim.to_csv(interim_path, index=False)
    processed.to_csv(processed_path, index=False)
    summary = {
        "source_url": EXCAPE_URL,
        "n_raw_rows": int(len(raw)),
        "n_interim_rows": int(len(interim)),
        "n_unique_molecules": int(len(processed)),
        "median_pIC50": float(processed["pIC50_median"].median()) if not processed.empty else None,
        "mean_support_score": float(processed["excape_support_score"].mean()) if not processed.empty else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved ExCAPE interim: {interim_path}")
    print(f"[OK] Saved ExCAPE reference: {processed_path}")
    print(f"[OK] Saved ExCAPE summary: {summary_path}")
    preview_cols = [
        "smiles_canonical",
        "pIC50_median",
        "n_measurements",
        "active_fraction",
        "excape_support_score",
    ]
    preview_cols = [column for column in preview_cols if column in processed.columns]
    print(processed[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
