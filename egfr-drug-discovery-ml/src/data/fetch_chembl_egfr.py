from __future__ import annotations

from pathlib import Path
import sys
import time
import pandas as pd
from chembl_webresource_client.new_client import new_client

from src.config import RAW_DIR, EGFR_TARGET_CHEMBL_ID


def fetch_egfr_ic50_raw(target_chembl_id: str = EGFR_TARGET_CHEMBL_ID, max_rows: int | None = None) -> pd.DataFrame:
    """
    Download raw activity rows for EGFR (IC50) from ChEMBL.
    Uses incremental iteration so we can show progress and avoid "silent hanging".
    """
    activity = new_client.activity

    # Build query (ChEMBL returns a generator-like iterable)
    res = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type="IC50",
    ).only([
        "activity_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_type",
        "pchembl_value",
        "document_chembl_id",
        "year",
    ])

    rows = []
    t0 = time.time()

    print(f"[INFO] Fetching activities for target={target_chembl_id} (IC50) ...", flush=True)

    # Iterate gradually so user sees progress
    for i, r in enumerate(res, start=1):
        rows.append(r)

        # progress every 500 rows
        if i % 500 == 0:
            dt = time.time() - t0
            print(f"[INFO] fetched {i} rows in {dt:.1f}s", flush=True)

        if max_rows is not None and i >= max_rows:
            print(f"[INFO] Reached max_rows={max_rows}. Stopping early.", flush=True)
            break

    print(f"[INFO] Done. Total rows fetched: {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    print("FETCH SCRIPT STARTED", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = RAW_DIR / "chembl_egfr_ic50_raw.csv"

    try:
        df = fetch_egfr_ic50_raw()
    except Exception as e:
        print("[ERROR] Failed to fetch from ChEMBL.", flush=True)
        print(repr(e), flush=True)
        sys.exit(1)

    df.to_csv(out_path, index=False)

    print(f"[OK] Saved raw dataset: {out_path}", flush=True)
    print(f"[OK] Rows: {len(df)}", flush=True)
    print(df.head(3), flush=True)


if __name__ == "__main__":
    main()