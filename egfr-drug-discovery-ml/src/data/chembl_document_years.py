from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

from src.config import INTERIM_DIR


DOCUMENT_YEAR_CACHE_PATH = INTERIM_DIR / "chembl_document_years.csv"


def _normalize_doc_id(value: object) -> str | None:
    if value is None:
        return None
    doc_id = str(value).strip()
    return doc_id or None


def _load_cache(cache_path: Path = DOCUMENT_YEAR_CACHE_PATH) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["document_chembl_id", "year"])
    cached = pd.read_csv(cache_path, low_memory=False)
    if cached.empty:
        return pd.DataFrame(columns=["document_chembl_id", "year"])
    cached["document_chembl_id"] = cached["document_chembl_id"].map(_normalize_doc_id)
    cached["year"] = pd.to_numeric(cached["year"], errors="coerce")
    cached = cached.dropna(subset=["document_chembl_id"]).drop_duplicates(subset=["document_chembl_id"], keep="last")
    return cached.reset_index(drop=True)


def _save_cache(cache_df: pd.DataFrame, cache_path: Path = DOCUMENT_YEAR_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = cache_df.dropna(subset=["document_chembl_id"]).drop_duplicates(subset=["document_chembl_id"], keep="last")
    cache_df = cache_df.sort_values("document_chembl_id").reset_index(drop=True)
    cache_df.to_csv(cache_path, index=False)


def fetch_document_year_map(
    document_ids: list[str],
    *,
    batch_size: int = 100,
    sleep_s: float = 0.05,
    cache_path: Path = DOCUMENT_YEAR_CACHE_PATH,
) -> dict[str, int | None]:
    normalized_ids = [_normalize_doc_id(value) for value in document_ids]
    target_ids = sorted({value for value in normalized_ids if value})
    if not target_ids:
        return {}

    cache_df = _load_cache(cache_path)
    cached_map = dict(
        zip(
            cache_df["document_chembl_id"].astype(str).tolist(),
            pd.to_numeric(cache_df["year"], errors="coerce").astype("Int64").tolist(),
        )
    )
    missing_ids = [doc_id for doc_id in target_ids if doc_id not in cached_map]
    if not missing_ids:
        return {doc_id: _coerce_nullable_int(cached_map.get(doc_id)) for doc_id in target_ids}

    document_client = new_client.document
    fetched_rows: list[dict[str, object]] = []
    for idx in range(0, len(missing_ids), batch_size):
        batch = missing_ids[idx : idx + batch_size]
        records = document_client.filter(document_chembl_id__in=batch).only(["document_chembl_id", "year"])
        for record in records:
            doc_id = _normalize_doc_id(record.get("document_chembl_id"))
            if doc_id is None:
                continue
            fetched_rows.append(
                {
                    "document_chembl_id": doc_id,
                    "year": _coerce_nullable_int(record.get("year")),
                }
            )
        time.sleep(sleep_s)

    fetched_df = pd.DataFrame(fetched_rows, columns=["document_chembl_id", "year"])
    if not fetched_df.empty:
        combined_cache = pd.concat([cache_df, fetched_df], ignore_index=True)
    else:
        combined_cache = cache_df
    for doc_id in missing_ids:
        if doc_id not in combined_cache.get("document_chembl_id", pd.Series(dtype=str)).astype(str).tolist():
            combined_cache = pd.concat(
                [
                    combined_cache,
                    pd.DataFrame([{"document_chembl_id": doc_id, "year": pd.NA}]),
                ],
                ignore_index=True,
            )
    _save_cache(combined_cache, cache_path=cache_path)

    refreshed_cache = _load_cache(cache_path)
    refreshed_map = dict(
        zip(
            refreshed_cache["document_chembl_id"].astype(str).tolist(),
            pd.to_numeric(refreshed_cache["year"], errors="coerce").astype("Int64").tolist(),
        )
    )
    return {doc_id: _coerce_nullable_int(refreshed_map.get(doc_id)) for doc_id in target_ids}


def _coerce_nullable_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
