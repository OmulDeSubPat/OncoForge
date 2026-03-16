from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.evaluation.random_split import random_split
from src.evaluation.scaffold_split import scaffold_split
from src.models.benchmark_model_families import evaluate_split, make_family_specs
from src.models.train_multiview_ensemble import build_features, evaluate_ensemble, make_model_bundles


def _feature_store(df: pd.DataFrame, smiles_col: str) -> dict[str, np.ndarray]:
    return build_features(df, smiles_col)


def _evaluate_multiview_once(
    feature_store: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    seed: int,
) -> dict[str, object]:
    metrics = evaluate_ensemble(make_model_bundles(), feature_store, y, train_idx, test_idx)
    return {
        "strategy": "multiview_ensemble",
        "model_family": "multiview_ensemble",
        "feature_set": "multiview",
        "split": split_name,
        "seed": int(seed),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "avg_uncertainty": float(metrics["avg_uncertainty"]),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
    }


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["strategy", "model_family", "feature_set", "split"], dropna=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_uncertainty=("avg_uncertainty", "mean"),
            std_uncertainty=("avg_uncertainty", "std"),
            runs=("seed", "nunique"),
        )
        .reset_index()
    )
    fill_cols = [column for column in summary.columns if column.startswith("std_")]
    if fill_cols:
        summary[fill_cols] = summary[fill_cols].fillna(0.0)
    summary["robustness_score"] = summary["mean_rmse"] + 0.35 * summary["std_rmse"]
    return summary.sort_values(
        ["split", "robustness_score", "mean_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _plot_robustness(summary: pd.DataFrame, out_dir) -> None:
    plot_df = summary[summary["split"] == "scaffold"].head(8).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(
        plot_df["model_family"],
        plot_df["mean_rmse"],
        yerr=plot_df["std_rmse"],
        fmt="o",
        color="#1d3557",
        ecolor="#457b9d",
        elinewidth=1.5,
        capsize=4,
    )
    ax.set_ylabel("Scaffold RMSE")
    ax.set_title("Repeated-Seed Robustness of Model Families")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "model_robustness_scaffold.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a repeated-seed robustness benchmark for model families and the multiview ensemble.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 42, 93],
        help="Random seeds used for repeated random/scaffold evaluations.",
    )
    args = parser.parse_args(argv)

    data_path = resolve_preferred_processed_dataset()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))
    smiles_col = "smiles_canonical"
    y = df["pIC50_median"].to_numpy(dtype=float)
    feature_store = _feature_store(df, smiles_col)
    family_specs = make_family_specs()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        random_train_df, random_test_df = random_split(df, test_size=0.2, seed=seed)
        scaffold_train_df, scaffold_test_df = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=seed)

        random_train_idx = random_train_df["_row_id"].to_numpy(dtype=int)
        random_test_idx = random_test_df["_row_id"].to_numpy(dtype=int)
        scaffold_train_idx = scaffold_train_df["_row_id"].to_numpy(dtype=int)
        scaffold_test_idx = scaffold_test_df["_row_id"].to_numpy(dtype=int)

        random_rows = evaluate_split(family_specs, feature_store, y, random_train_idx, random_test_idx, "random")
        scaffold_rows = evaluate_split(family_specs, feature_store, y, scaffold_train_idx, scaffold_test_idx, "scaffold")
        for row in random_rows + scaffold_rows:
            row["strategy"] = "model_family"
            row["seed"] = int(seed)
            row["avg_uncertainty"] = np.nan
            rows.append(row)

        rows.append(_evaluate_multiview_once(feature_store, y, random_train_idx, random_test_idx, "random", seed))
        rows.append(_evaluate_multiview_once(feature_store, y, scaffold_train_idx, scaffold_test_idx, "scaffold", seed))

    raw = pd.DataFrame(rows)
    summary = _summarize(raw)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reports_dir / "model_robustness_raw.csv"
    summary_path = reports_dir / "model_robustness_summary.csv"
    json_path = reports_dir / "model_robustness_summary.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_label_from_path(data_path),
                "dataset_path": str(data_path),
                "seeds": [int(seed) for seed in args.seeds],
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_robustness(summary, reports_dir)

    print(f"[OK] Saved robustness raw results: {raw_path}")
    print(f"[OK] Saved robustness summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
