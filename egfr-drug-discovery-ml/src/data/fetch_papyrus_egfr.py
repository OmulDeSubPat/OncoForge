from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import requests

from src.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from src.utils.chem import canonicalize_smiles


PAPYRUS_DIR = EXTERNAL_DIR / "papyrus"
PAPYRUS_FILENAME = "05.6++_combined_set_without_stereochemistry.tsv.xz"
PAPYRUS_URL = f"https://zenodo.org/records/13817795/files/{PAPYRUS_FILENAME}?download=1"
PAPYRUS_TARGET_ACCESSION = "P00533"
PAPYRUS_CHUNK_SIZE = 100_000


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


def _papyrus_pic50_to_ic50_nm(value: float | int | None) -> float | None:
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
        chunksize=PAPYRUS_CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in chunk_iter:
        accession = chunk.get("accession", pd.Series("", index=chunk.index)).astype(str)
        target_id = chunk.get("target_id", pd.Series("", index=chunk.index)).astype(str)
        tid = chunk.get("TID", pd.Series("", index=chunk.index)).astype(str)
        mask = (
            accession.eq(PAPYRUS_TARGET_ACCESSION)
            | target_id.str.contains(PAPYRUS_TARGET_ACCESSION, case=False, na=False)
            | tid.str.contains("EGFR", case=False, na=False)
        )
        if mask.any():
            yield chunk.loc[mask].copy()


def _coerce_papyrus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out["source_dataset"] = "papyrus"
    out["source_record_id"] = out["Activity_ID"].astype(str)
    out["smiles_canonical"] = out["SMILES"].map(canonicalize_smiles)
    out["pIC50"] = pd.to_numeric(out.get("pchembl_value_Mean"), errors="coerce")
    out.loc[out["pIC50"].isna(), "pIC50"] = pd.to_numeric(out.get("pchembl_value_Median"), errors="coerce")
    out.loc[out["pIC50"].isna(), "pIC50"] = pd.to_numeric(out.get("pchembl_value"), errors="coerce")
    out["ic50_nm"] = out["pIC50"].map(_papyrus_pic50_to_ic50_nm)
    out["year"] = pd.to_numeric(out.get("Year"), errors="coerce")
    out["papyrus_quality"] = out.get("Quality", pd.Series("", index=out.index)).astype(str)
    out["papyrus_source"] = out.get("source", pd.Series("", index=out.index)).astype(str)
    out["papyrus_relation"] = out.get("relation", pd.Series("", index=out.index)).astype(str)
    out["papyrus_protein_type"] = out.get("Protein_Type", pd.Series("", index=out.index)).astype(str)
    out["papyrus_doc_id"] = out.get("doc_id", pd.Series("", index=out.index)).astype(str)
    out = out[
        out["smiles_canonical"].notna()
        & out["pIC50"].notna()
        & out["ic50_nm"].notna()
        & out["papyrus_quality"].isin(["High", "Medium"])
        & out["papyrus_relation"].isin(["=", "~"])
    ].copy()

    interim = out[
        [
            "source_dataset",
            "source_record_id",
            "smiles_canonical",
            "ic50_nm",
            "pIC50",
            "year",
            "papyrus_quality",
            "papyrus_source",
            "papyrus_relation",
            "papyrus_protein_type",
            "papyrus_doc_id",
            "target_id",
            "accession",
        ]
    ].reset_index(drop=True)

    processed = (
        interim.groupby("smiles_canonical", as_index=False)
        .agg(
            pIC50_median=("pIC50", "median"),
            ic50_nm_median=("ic50_nm", "median"),
            n_measurements=("pIC50", "size"),
            papyrus_quality_best=("papyrus_quality", lambda values: "High" if "High" in set(values) else "Medium"),
            papyrus_sources=("papyrus_source", lambda values: ";".join(sorted(set(str(value) for value in values if value)))),
            papyrus_doc_count=("papyrus_doc_id", lambda values: int(pd.Series(values).replace("", pd.NA).dropna().nunique())),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .sort_values(["pIC50_median", "n_measurements"], ascending=[False, False])
        .reset_index(drop=True)
    )
    processed["source_dataset"] = "papyrus"
    processed["papyrus_support_score"] = (
        0.45 * processed["pIC50_median"].map(lambda value: max(0.0, min(1.0, (float(value) - 5.5) / 3.5)))
        + 0.25 * processed["n_measurements"].map(lambda value: min(1.0, math.log1p(float(value)) / math.log(8.0)))
        + 0.15 * processed["papyrus_doc_count"].map(lambda value: min(1.0, math.log1p(float(value)) / math.log(6.0)))
        + 0.15 * processed["papyrus_quality_best"].eq("High").astype(float)
    ).clip(lower=0.0, upper=1.0)
    return interim, processed


def ensure_papyrus_reference() -> Path:
    processed_path = PROCESSED_DIR / "papyrus_egfr_reference.csv"
    if processed_path.exists():
        return processed_path
    main()
    return processed_path


def main() -> None:
    PAPYRUS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = _download_if_missing(PAPYRUS_URL, PAPYRUS_DIR / PAPYRUS_FILENAME)
    frames = list(_iter_target_chunks(dataset_path))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    interim, processed = _coerce_papyrus(raw)

    interim_path = INTERIM_DIR / "papyrus_egfr_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "papyrus_egfr_reference.csv"
    summary_path = PROCESSED_DIR / "papyrus_egfr_reference.summary.json"

    interim.to_csv(interim_path, index=False)
    processed.to_csv(processed_path, index=False)
    summary = {
        "source_url": PAPYRUS_URL,
        "n_raw_rows": int(len(raw)),
        "n_interim_rows": int(len(interim)),
        "n_unique_molecules": int(len(processed)),
        "median_pIC50": float(processed["pIC50_median"].median()) if not processed.empty else None,
        "mean_support_score": float(processed["papyrus_support_score"].mean()) if not processed.empty else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved Papyrus interim: {interim_path}")
    print(f"[OK] Saved Papyrus reference: {processed_path}")
    print(f"[OK] Saved Papyrus summary: {summary_path}")
    preview_cols = [
        "smiles_canonical",
        "pIC50_median",
        "n_measurements",
        "papyrus_quality_best",
        "papyrus_support_score",
    ]
    preview_cols = [column for column in preview_cols if column in processed.columns]
    print(processed[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
