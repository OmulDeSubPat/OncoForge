from __future__ import annotations

from typing import Tuple

import pandas as pd


def _value_counts_as_dict(df: pd.DataFrame, source_col: str | None) -> dict[str, int]:
    if source_col is None or source_col not in df.columns:
        return {}
    values = df[source_col].fillna("<missing>").astype(str)
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}


def _source_imbalance(train_df: pd.DataFrame, test_df: pd.DataFrame, source_col: str | None) -> float:
    train_counts = _value_counts_as_dict(train_df, source_col)
    test_counts = _value_counts_as_dict(test_df, source_col)
    labels = sorted(set(train_counts) | set(test_counts))
    if not labels:
        return 0.0

    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    if train_total == 0 or test_total == 0:
        return float("inf")

    return 0.5 * sum(
        abs(train_counts.get(label, 0) / train_total - test_counts.get(label, 0) / test_total)
        for label in labels
    )


def temporal_split(
    df: pd.DataFrame,
    year_col: str = "year_max",
    year_min_col: str = "year_min",
    test_size: float = 0.2,
    min_rows: int = 250,
    min_train_rows: int = 250,
    min_test_rows: int = 100,
    source_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Split by year so that newer measurements are held out.
    Rows without year metadata are excluded from this evaluation.

    When a source column is available, the cutoff is chosen to keep the
    source composition reasonably balanced across train and test.
    """
    if year_col not in df.columns:
        raise ValueError(f"Missing year_col='{year_col}' in df.columns")

    dated = df.copy()
    dated[year_col] = pd.to_numeric(dated[year_col], errors="coerce")
    if year_min_col in dated.columns:
        dated[year_min_col] = pd.to_numeric(dated[year_min_col], errors="coerce")

    if year_min_col in dated.columns and dated[year_min_col].notna().any():
        ranged = dated[dated[year_col].notna() & dated[year_min_col].notna()].copy()
        if len(ranged) >= min_rows:
            ranged[year_col] = ranged[year_col].astype(int)
            ranged[year_min_col] = ranged[year_min_col].astype(int)
            target_test = max(1, int(len(ranged) * test_size))
            candidate_splits: list[tuple[float, int, float, int, int]] = []
            for year in sorted(ranged[year_col].unique()):
                train_mask = ranged[year_col] < year
                test_mask = ranged[year_min_col] >= year
                spanning_mask = (ranged[year_min_col] < year) & (ranged[year_col] >= year)
                train_count = int(train_mask.sum())
                test_count = int(test_mask.sum())
                spanning_count = int(spanning_mask.sum())
                if train_count >= min_train_rows and test_count >= min_test_rows:
                    train_subset = ranged.loc[train_mask]
                    test_subset = ranged.loc[test_mask]
                    source_penalty = _source_imbalance(train_subset, test_subset, source_col)
                    candidate_splits.append(
                        (
                            float(abs(test_count - target_test)),
                            spanning_count,
                            float(source_penalty),
                            int(year),
                            test_count,
                        )
                    )

            if candidate_splits:
                _, spanning_rows, source_penalty, cutoff_year, _ = min(
                    candidate_splits,
                    key=lambda item: (item[0], item[1], item[2], item[3]),
                )
                train_df = ranged[ranged[year_col] < cutoff_year].reset_index(drop=True)
                test_df = ranged[ranged[year_min_col] >= cutoff_year].reset_index(drop=True)
                metadata: dict[str, object] = {
                    "strategy": "non_overlapping_year_ranges",
                    "year_col": year_col,
                    "year_min_col": year_min_col,
                    "source_col": source_col,
                    "cutoff_year": cutoff_year,
                    "dated_rows": int(len(ranged)),
                    "undated_rows": int(len(df) - len(ranged)),
                    "excluded_spanning_rows": int(spanning_rows),
                    "source_imbalance": float(source_penalty),
                    "train_source_counts": _value_counts_as_dict(train_df, source_col),
                    "test_source_counts": _value_counts_as_dict(test_df, source_col),
                    "train_size": int(len(train_df)),
                    "test_size": int(len(test_df)),
                }
                return train_df, test_df, metadata

    dated = dated[dated[year_col].notna()].copy()
    if len(dated) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows with non-null {year_col} for temporal split; found {len(dated)}."
        )

    dated[year_col] = dated[year_col].astype(int)
    dated = dated.sort_values(year_col).reset_index(drop=True)
    target_test = max(1, int(len(dated) * test_size))

    candidate_splits = []
    for year in sorted(dated[year_col].unique()):
        test_count = int((dated[year_col] >= year).sum())
        train_count = int((dated[year_col] < year).sum())
        if train_count >= min_train_rows and test_count >= min_test_rows:
            train_subset = dated[dated[year_col] < year]
            test_subset = dated[dated[year_col] >= year]
            source_penalty = _source_imbalance(train_subset, test_subset, source_col)
            candidate_splits.append(
                (
                    float(abs(test_count - target_test)),
                    float(source_penalty),
                    int(year),
                    test_count,
                )
            )

    if not candidate_splits:
        raise ValueError(
            "Unable to determine a temporal cutoff year with enough train/test support. "
            f"Need at least train={min_train_rows}, test={min_test_rows}."
        )

    _, source_penalty, cutoff_year, _ = min(candidate_splits, key=lambda item: (item[0], item[1], item[2]))

    train_df = dated[dated[year_col] < cutoff_year].reset_index(drop=True)
    test_df = dated[dated[year_col] >= cutoff_year].reset_index(drop=True)
    metadata = {
        "strategy": "single_year_cutoff",
        "year_col": year_col,
        "year_min_col": year_min_col,
        "source_col": source_col,
        "cutoff_year": cutoff_year,
        "dated_rows": int(len(dated)),
        "undated_rows": int(len(df) - len(dated)),
        "source_imbalance": float(source_penalty),
        "train_source_counts": _value_counts_as_dict(train_df, source_col),
        "test_source_counts": _value_counts_as_dict(test_df, source_col),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }
    return train_df, test_df, metadata
