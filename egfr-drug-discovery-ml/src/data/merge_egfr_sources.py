from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset artifact: {path}")
    return pd.read_csv(path, low_memory=False)


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


def _default_year_provenance(source_name: str) -> tuple[str, float]:
    if source_name == "chembl":
        return "document_chembl_id", 0.95
    if source_name == "bindingdb_articles":
        return "publication_date", 0.95
    if source_name == "papyrus":
        return "Year", 0.9
    if source_name.startswith("excape"):
        return "year", 0.85
    return "year", 0.8


def _coerce_interim_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    out["source_dataset"] = out.get("source_dataset", source_name)
    if "source_record_id" not in out.columns:
        if source_name == "chembl":
            out["source_record_id"] = out.get("activity_id")
        else:
            out["source_record_id"] = pd.RangeIndex(1, len(out) + 1, name="source_record_id")

    if "ic50_nm" not in out.columns and "standard_value" in out.columns:
        out["ic50_nm"] = pd.to_numeric(out["standard_value"], errors="coerce")

    if "year" not in out.columns:
        if "year_min" in out.columns:
            out["year"] = pd.to_numeric(out["year_min"], errors="coerce")
            year_source_field = "year_min"
        else:
            out["year"] = pd.NA
    out["year"] = pd.to_numeric(out["year"], errors="coerce")

    year_source_field, default_year_confidence = _default_year_provenance(source_name)
    if "year_source" not in out.columns:
        out["year_source"] = pd.NA
    if "year_confidence" not in out.columns:
        out["year_confidence"] = pd.NA
    out.loc[out["year"].notna() & out["year_source"].isna(), "year_source"] = year_source_field
    out.loc[out["year"].notna() & out["year_confidence"].isna(), "year_confidence"] = default_year_confidence
    out.loc[out["year"].isna() & out["year_source"].isna(), "year_source"] = "missing"
    out["year_confidence"] = pd.to_numeric(out["year_confidence"], errors="coerce").fillna(0.0)

    keep_cols = [
        "source_dataset",
        "source_record_id",
        "smiles_canonical",
        "ic50_nm",
        "pIC50",
        "year",
        "year_source",
        "year_confidence",
    ]
    if "molecule_chembl_id" in out.columns:
        keep_cols.append("molecule_chembl_id")
    if "bindingdb_reactant_set_id" in out.columns:
        keep_cols.append("bindingdb_reactant_set_id")
    if "document_chembl_id" in out.columns:
        keep_cols.append("document_chembl_id")
    for column in [
        "temporal_year_min",
        "temporal_year_max",
        "temporal_year_coverage_rate",
        "temporal_year_source_count",
        "temporal_year_sources",
    ]:
        if column in out.columns:
            keep_cols.append(column)

    available = [column for column in keep_cols if column in out.columns]
    out = out[available].copy()
    out["ic50_nm"] = pd.to_numeric(out["ic50_nm"], errors="coerce")
    out["pIC50"] = pd.to_numeric(out["pIC50"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    if "year_confidence" in out.columns:
        out["year_confidence"] = pd.to_numeric(out["year_confidence"], errors="coerce").fillna(0.0)
    out = out[out["smiles_canonical"].notna() & out["pIC50"].notna() & out["ic50_nm"].notna()].copy()
    return out.reset_index(drop=True)


def main() -> None:
    required_sources = [
        ("chembl", INTERIM_DIR / "chembl_egfr_ic50_interim.csv"),
        ("bindingdb_articles", INTERIM_DIR / "bindingdb_egfr_ic50_interim.csv"),
    ]
    optional_sources = [
        ("papyrus", INTERIM_DIR / "papyrus_egfr_ic50_interim.csv"),
        ("excape", INTERIM_DIR / "excape_egfr_ic50_interim.csv"),
    ]

    standardized_frames = [
        _coerce_interim_schema(_load_required_csv(path), source_name)
        for source_name, path in required_sources
    ]
    included_optional: list[str] = []
    for source_name, path in optional_sources:
        if not path.exists():
            continue
        standardized_frames.append(_coerce_interim_schema(pd.read_csv(path, low_memory=False), source_name))
        included_optional.append(source_name)

    interim_df = pd.concat(standardized_frames, ignore_index=True)
    interim_df = interim_df.drop_duplicates(
        subset=["source_dataset", "source_record_id", "smiles_canonical", "ic50_nm"]
    ).reset_index(drop=True)

    processed_df = (
        interim_df.groupby("smiles_canonical", as_index=False)
        .agg(
            ic50_nm_median=("ic50_nm", "median"),
            pIC50_median=("pIC50", "median"),
            n_measurements=("pIC50", "size"),
            n_sources=("source_dataset", "nunique"),
            source_datasets=("source_dataset", _join_unique_non_missing),
            year_source_count=("year_source", _count_unique_non_missing),
            year_sources=("year_source", _join_unique_non_missing),
            year_confidence_mean=("year_confidence", "mean"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            temporal_year_min=("year", "min"),
            temporal_year_max=("year", "max"),
            temporal_year_coverage_rate=("year", lambda values: float(values.notna().mean())),
        )
        .sort_values("pIC50_median", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "interim_rows": int(len(interim_df)),
        "unique_molecules": int(len(processed_df)),
        "rows_with_year": int(interim_df["year"].notna().sum()),
        "rows_with_year_source": int(interim_df["year_source"].notna().sum()) if "year_source" in interim_df.columns else 0,
        "mean_year_confidence": float(interim_df["year_confidence"].mean()) if "year_confidence" in interim_df.columns and not interim_df.empty else None,
        "chembl_rows": int((interim_df["source_dataset"] == "chembl").sum()),
        "bindingdb_rows": int((interim_df["source_dataset"] == "bindingdb_articles").sum()),
        "papyrus_rows": int((interim_df["source_dataset"] == "papyrus").sum()),
        "excape_rows": int(interim_df["source_dataset"].astype(str).str.startswith("excape").sum()),
        "optional_sources_included": included_optional,
        "molecules_with_multiple_sources": int((processed_df["n_sources"] > 1).sum()),
    }

    interim_path = INTERIM_DIR / "egfr_multisource_ic50_interim.csv"
    processed_path = PROCESSED_DIR / "egfr_multisource_ic50_clean.csv"
    summary_path = PROCESSED_DIR / "egfr_multisource_summary.json"

    interim_df.to_csv(interim_path, index=False)
    processed_df.to_csv(processed_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Multisource interim saved:   {interim_path}  (rows={len(interim_df)})")
    print(f"[OK] Multisource processed saved: {processed_path} (molecules={len(processed_df)})")
    print(f"[OK] Multisource summary saved:   {summary_path}")
    print(processed_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
