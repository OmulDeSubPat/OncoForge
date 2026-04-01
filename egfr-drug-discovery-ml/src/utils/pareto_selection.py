from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = float(values.min()) if not values.empty else 0.0
    high = float(values.max()) if not values.empty else 0.0
    if math.isclose(low, high):
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((values - low) / (high - low)).clip(lower=0.0, upper=1.0)


def _dominates(
    candidate_a: int,
    candidate_b: int,
    maximize_matrix: np.ndarray,
    minimize_matrix: np.ndarray,
) -> bool:
    better_or_equal = True
    strictly_better = False

    if maximize_matrix.size:
        a_vals = maximize_matrix[candidate_a]
        b_vals = maximize_matrix[candidate_b]
        if np.any(a_vals < b_vals):
            better_or_equal = False
        if np.any(a_vals > b_vals):
            strictly_better = True

    if minimize_matrix.size and better_or_equal:
        a_vals = minimize_matrix[candidate_a]
        b_vals = minimize_matrix[candidate_b]
        if np.any(a_vals > b_vals):
            better_or_equal = False
        if np.any(a_vals < b_vals):
            strictly_better = True

    return bool(better_or_equal and strictly_better)


def _crowding_distance(front_indices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(front_indices) == 0:
        return np.asarray([], dtype=float)
    if len(front_indices) <= 2 or matrix.size == 0:
        return np.ones(len(front_indices), dtype=float)

    distances = np.zeros(len(front_indices), dtype=float)
    front_values = matrix[front_indices]
    n_objectives = front_values.shape[1]
    for objective_idx in range(n_objectives):
        objective_values = front_values[:, objective_idx]
        order = np.argsort(objective_values)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        min_value = float(objective_values[order[0]])
        max_value = float(objective_values[order[-1]])
        if math.isclose(min_value, max_value):
            continue
        scale = max_value - min_value
        for rank_idx in range(1, len(order) - 1):
            prev_value = float(objective_values[order[rank_idx - 1]])
            next_value = float(objective_values[order[rank_idx + 1]])
            distances[order[rank_idx]] += (next_value - prev_value) / scale

    if np.isinf(distances).any():
        finite = distances[np.isfinite(distances)]
        baseline = float(finite.max()) if finite.size else 1.0
        distances[np.isinf(distances)] = max(1.0, baseline + 1.0)
    if distances.max() > 0:
        distances = distances / distances.max()
    return distances


def add_pareto_front_columns(
    df: pd.DataFrame,
    *,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
    prefix: str = "pareto",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    maximize = maximize or []
    minimize = minimize or []
    usable_maximize = [column for column in maximize if column in df.columns]
    usable_minimize = [column for column in minimize if column in df.columns]
    if not usable_maximize and not usable_minimize:
        out = df.copy()
        out[f"{prefix}_front_rank"] = 1
        out[f"{prefix}_crowding_score"] = 0.5
        out[f"{prefix}_is_front"] = True
        out[f"{prefix}_priority_bonus"] = 1.0
        return out

    maximize_matrix = (
        np.column_stack([_series(df, column, 0.0).to_numpy(dtype=float) for column in usable_maximize])
        if usable_maximize
        else np.empty((len(df), 0))
    )
    minimize_matrix = (
        np.column_stack([_series(df, column, 0.0).to_numpy(dtype=float) for column in usable_minimize])
        if usable_minimize
        else np.empty((len(df), 0))
    )

    out = df.copy()
    indices = np.arange(len(out))
    front_rank = np.zeros(len(out), dtype=int)
    crowding = np.zeros(len(out), dtype=float)
    remaining = indices.copy()
    current_rank = 1
    objective_matrix = np.column_stack(
        [_normalize(_series(out, column, 0.0)).to_numpy(dtype=float) for column in usable_maximize]
        + [_normalize(1.0 - _series(out, column, 0.0)).to_numpy(dtype=float) for column in usable_minimize]
    )

    while remaining.size:
        front: list[int] = []
        for candidate_idx in remaining:
            dominated = False
            for challenger_idx in remaining:
                if challenger_idx == candidate_idx:
                    continue
                if _dominates(challenger_idx, candidate_idx, maximize_matrix, minimize_matrix):
                    dominated = True
                    break
            if not dominated:
                front.append(int(candidate_idx))
        front_indices = np.asarray(front, dtype=int)
        front_rank[front_indices] = current_rank
        crowding[front_indices] = _crowding_distance(front_indices, objective_matrix)
        remaining = np.asarray([idx for idx in remaining if idx not in set(front)], dtype=int)
        current_rank += 1

    priority_bonus = (1.0 / np.maximum(front_rank, 1)).astype(float) + 0.25 * crowding
    out[f"{prefix}_front_rank"] = front_rank
    out[f"{prefix}_crowding_score"] = crowding
    out[f"{prefix}_is_front"] = front_rank == 1
    out[f"{prefix}_priority_bonus"] = priority_bonus
    return out
