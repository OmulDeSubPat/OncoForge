from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    year_col: str = "year_max",
    test_size: float = 0.2,
    min_rows: int = 250,
    min_train_rows: int = 250,
    min_test_rows: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """
    Split by year so that newer measurements are held out.
    Rows without year metadata are excluded from this evaluation.
    """
    if year_col not in df.columns:
        raise ValueError(f"Missing year_col='{year_col}' in df.columns")

    dated = df[df[year_col].notna()].copy()
    if len(dated) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows with non-null {year_col} for temporal split; found {len(dated)}."
        )

    dated[year_col] = dated[year_col].astype(int)
    dated = dated.sort_values(year_col).reset_index(drop=True)
    target_test = max(1, int(len(dated) * test_size))

    candidate_splits: list[tuple[int, int, int]] = []
    for year in sorted(dated[year_col].unique()):
        test_count = int((dated[year_col] >= year).sum())
        train_count = int((dated[year_col] < year).sum())
        if train_count >= min_train_rows and test_count >= min_test_rows:
            candidate_splits.append((abs(test_count - target_test), int(year), test_count))

    if not candidate_splits:
        raise ValueError(
            "Unable to determine a temporal cutoff year with enough train/test support. "
            f"Need at least train={min_train_rows}, test={min_test_rows}."
        )

    _, cutoff_year, _ = min(candidate_splits, key=lambda item: (item[0], item[1]))

    train_df = dated[dated[year_col] < cutoff_year].reset_index(drop=True)
    test_df = dated[dated[year_col] >= cutoff_year].reset_index(drop=True)
    metadata = {
        "year_col": year_col,
        "cutoff_year": cutoff_year,
        "dated_rows": int(len(dated)),
        "undated_rows": int(len(df) - len(dated)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }
    return train_df, test_df, metadata
