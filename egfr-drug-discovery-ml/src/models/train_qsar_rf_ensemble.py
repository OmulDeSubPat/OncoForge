from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.evaluation.random_split import random_split
from src.evaluation.scaffold_split import scaffold_split
from src.features.featurize_ecfp import ecfp_from_smiles


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict[str, Any]


def featurize_df(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    return np.vstack([ecfp_from_smiles(s) for s in df[smiles_col].tolist()]).astype(np.float32)


def build_model(spec: ModelSpec):
    if spec.family == "rf":
        return RandomForestRegressor(**spec.params)
    if spec.family == "et":
        return ExtraTreesRegressor(**spec.params)
    if spec.family == "hgb":
        return HistGradientBoostingRegressor(**spec.params)
    raise ValueError(f"Unsupported model family: {spec.family}")


def make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="rf_sqrt_bootstrap",
            family="rf",
            params={
                "n_estimators": 400,
                "random_state": 11,
                "n_jobs": -1,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": 0.85,
            },
        ),
        ModelSpec(
            name="rf_regularized_leaf2",
            family="rf",
            params={
                "n_estimators": 500,
                "random_state": 29,
                "n_jobs": -1,
                "max_features": 0.20,
                "bootstrap": True,
                "max_samples": 0.85,
                "min_samples_leaf": 2,
            },
        ),
        ModelSpec(
            name="et_sqrt_dense",
            family="et",
            params={
                "n_estimators": 700,
                "random_state": 17,
                "n_jobs": -1,
                "max_features": "sqrt",
            },
        ),
        ModelSpec(
            name="et_regularized_leaf2",
            family="et",
            params={
                "n_estimators": 800,
                "random_state": 41,
                "n_jobs": -1,
                "max_features": 0.18,
                "min_samples_leaf": 2,
            },
        ),
        ModelSpec(
            name="hgb_compact",
            family="hgb",
            params={
                "learning_rate": 0.05,
                "max_iter": 500,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 20,
                "l2_regularization": 0.05,
                "early_stopping": False,
                "random_state": 42,
            },
        ),
    ]


def compute_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def fit_predict(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    model = build_model(spec)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def evaluate_specs(
    specs: list[ModelSpec],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    for spec in specs:
        pred = fit_predict(spec, x_train, y_train, x_test)
        predictions[spec.name] = pred
        metrics = compute_metrics(y_test, pred)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                **metrics,
            }
        )

    return rows, predictions


def select_ensemble_members(benchmark_df: pd.DataFrame, max_models: int = 4) -> list[str]:
    ordered = benchmark_df.sort_values(
        ["scaffold_rmse", "scaffold_r2", "random_rmse", "random_r2"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)

    selected_names: list[str] = []
    selected_families: set[str] = set()

    best_by_family = ordered.groupby("family", sort=False).head(1)
    for _, row in best_by_family.iterrows():
        if row["name"] not in selected_names and len(selected_names) < max_models:
            selected_names.append(str(row["name"]))
            selected_families.add(str(row["family"]))

    for _, row in ordered.iterrows():
        if row["name"] in selected_names:
            continue
        if len(selected_names) >= max_models:
            break
        selected_names.append(str(row["name"]))
        selected_families.add(str(row["family"]))

    return selected_names


def ensemble_metrics(
    selected_names: list[str],
    prediction_store: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    pred_matrix = np.vstack([prediction_store[name] for name in selected_names]).astype(float)
    pred_mean = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)
    metrics = compute_metrics(y_true, pred_mean)
    metrics["avg_uncertainty"] = float(pred_std.mean())
    metrics["n_models"] = int(len(selected_names))
    return metrics


def main():
    data_path = resolve_preferred_processed_dataset()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {data_path}\n"
            "Run: python -m src.data.fetch_chembl_egfr && python -m src.data.clean_egfr_ic50"
        )

    df = pd.read_csv(data_path).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"
    specs = make_model_specs()

    print("[INFO] Building fingerprint matrix...")
    x_full = featurize_df(df, smiles_col)
    y_full = df[y_col].values.astype(float)

    random_train_df, random_test_df = random_split(df, test_size=0.2, seed=42)
    scaffold_train_df, scaffold_test_df = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=42)

    random_train_idx = random_train_df["_row_id"].to_numpy(dtype=int)
    random_test_idx = random_test_df["_row_id"].to_numpy(dtype=int)
    scaffold_train_idx = scaffold_train_df["_row_id"].to_numpy(dtype=int)
    scaffold_test_idx = scaffold_test_df["_row_id"].to_numpy(dtype=int)

    x_random_train = x_full[random_train_idx]
    y_random_train = y_full[random_train_idx]
    x_random_test = x_full[random_test_idx]
    y_random_test = y_full[random_test_idx]

    x_scaffold_train = x_full[scaffold_train_idx]
    y_scaffold_train = y_full[scaffold_train_idx]
    x_scaffold_test = x_full[scaffold_test_idx]
    y_scaffold_test = y_full[scaffold_test_idx]

    print("[INFO] Evaluating candidate models on random split...")
    random_rows, random_predictions = evaluate_specs(
        specs,
        x_random_train,
        y_random_train,
        x_random_test,
        y_random_test,
    )

    print("[INFO] Evaluating candidate models on scaffold split...")
    scaffold_rows, scaffold_predictions = evaluate_specs(
        specs,
        x_scaffold_train,
        y_scaffold_train,
        x_scaffold_test,
        y_scaffold_test,
    )

    benchmark_df = pd.DataFrame(random_rows).rename(
        columns={"mae": "random_mae", "rmse": "random_rmse", "r2": "random_r2"}
    )
    benchmark_df = benchmark_df.merge(
        pd.DataFrame(scaffold_rows).rename(
            columns={"mae": "scaffold_mae", "rmse": "scaffold_rmse", "r2": "scaffold_r2"}
        ),
        on=["name", "family"],
        how="inner",
    )

    selected_names = select_ensemble_members(benchmark_df, max_models=4)
    benchmark_df["selected"] = benchmark_df["name"].isin(selected_names)
    benchmark_df = benchmark_df.sort_values(
        ["selected", "scaffold_rmse", "scaffold_r2", "random_rmse"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    selected_specs = [spec for spec in specs if spec.name in selected_names]
    random_metrics = ensemble_metrics(selected_names, random_predictions, y_random_test)
    scaffold_metrics = ensemble_metrics(selected_names, scaffold_predictions, y_scaffold_test)

    print("[INFO] Selected ensemble members:", ", ".join(selected_names))
    print("[INFO] Random split ensemble:", random_metrics)
    print("[INFO] Scaffold split ensemble:", scaffold_metrics)

    print("[INFO] Retraining selected ensemble on full dataset...")
    models = []
    for spec in selected_specs:
        model = build_model(spec)
        model.fit(x_full, y_full)
        models.append(model)

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "qsar_rf_ensemble.pkl"
    joblib.dump(models, model_path, compress=3)

    metadata_path = out_dir / "qsar_rf_ensemble_metadata.json"
    metadata = {
        "selected_models": [asdict(spec) for spec in selected_specs],
        "all_candidate_models": [asdict(spec) for spec in specs],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = reports_dir / "model_candidate_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    metrics = {
        "dataset_size": int(len(df)),
        "dataset_name": dataset_label_from_path(data_path),
        "dataset_path": str(data_path),
        "smiles_column": smiles_col,
        "target_column": y_col,
        "n_models": len(models),
        "selected_models": selected_names,
        "random_split": {
            **random_metrics,
            "train_size": int(len(random_train_idx)),
            "test_size": int(len(random_test_idx)),
        },
        "scaffold_split": {
            **scaffold_metrics,
            "train_size": int(len(scaffold_train_idx)),
            "test_size": int(len(scaffold_test_idx)),
        },
    }
    metrics_path = reports_dir / "model_performance_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved ensemble model: {model_path}")
    print(f"[OK] Saved ensemble metadata: {metadata_path}")
    print(f"[OK] Saved candidate benchmark: {benchmark_path}")
    print(f"[OK] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
