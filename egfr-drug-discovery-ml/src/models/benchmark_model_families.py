from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.evaluation.random_split import random_split
from src.evaluation.scaffold_split import scaffold_split
from src.features.descriptor_features import DESCRIPTOR_NAMES, descriptor_vector_from_smiles
from src.features.featurize_ecfp import ecfp_from_smiles
from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class FamilySpec:
    name: str
    feature_set: str
    model_factory: callable


def featurize_df(df: pd.DataFrame, smiles_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ecfp = np.vstack([ecfp_from_smiles(smiles) for smiles in df[smiles_col].tolist()]).astype(np.float32)
    descriptors = np.vstack([descriptor_vector_from_smiles(smiles) for smiles in df[smiles_col].tolist()]).astype(np.float32)
    hybrid = np.hstack([ecfp, descriptors]).astype(np.float32)
    return ecfp, descriptors, hybrid


def metrics_dict(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def make_family_specs() -> list[FamilySpec]:
    return [
        FamilySpec(
            name="ridge_descriptors",
            feature_set="descriptors",
            model_factory=lambda: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        FamilySpec(
            name="knn_descriptors",
            feature_set="descriptors",
            model_factory=lambda: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
                ]
            ),
        ),
        FamilySpec(
            name="rf_descriptors",
            feature_set="descriptors",
            model_factory=lambda: RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
            ),
        ),
        FamilySpec(
            name="et_ecfp",
            feature_set="ecfp",
            model_factory=lambda: ExtraTreesRegressor(
                n_estimators=700,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
            ),
        ),
        FamilySpec(
            name="hgb_ecfp",
            feature_set="ecfp",
            model_factory=lambda: HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.05,
                early_stopping=False,
                random_state=42,
            ),
        ),
        FamilySpec(
            name="rf_hybrid",
            feature_set="hybrid",
            model_factory=lambda: RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                max_features=0.18,
                bootstrap=True,
                max_samples=0.85,
            ),
        ),
        FamilySpec(
            name="mlp_hybrid",
            feature_set="hybrid",
            model_factory=lambda: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(256, 96),
                            activation="relu",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=200,
                            early_stopping=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def pick_feature_matrix(feature_store: dict[str, np.ndarray], spec: FamilySpec) -> np.ndarray:
    return feature_store[spec.feature_set]


def evaluate_split(
    specs: list[FamilySpec],
    feature_store: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_train = y[train_idx]
    y_test = y[test_idx]

    for spec in specs:
        x = pick_feature_matrix(feature_store, spec)
        x_train = x[train_idx]
        x_test = x[test_idx]
        model = spec.model_factory()
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        metrics = metrics_dict(y_test, pred)
        rows.append(
            {
                "model_family": spec.name,
                "feature_set": spec.feature_set,
                "split": split_name,
                **metrics,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
            }
        )
    return rows


def main() -> None:
    data_path = resolve_preferred_processed_dataset()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"
    y = df[y_col].to_numpy(dtype=float)

    print("[INFO] Building feature matrices for model-family benchmark...")
    ecfp, descriptors, hybrid = featurize_df(df, smiles_col)
    feature_store = {
        "ecfp": ecfp,
        "descriptors": descriptors,
        "hybrid": hybrid,
    }

    specs = make_family_specs()
    random_train_df, random_test_df = random_split(df, test_size=0.2, seed=42)
    scaffold_train_df, scaffold_test_df = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=42)

    rows = []
    rows.extend(
        evaluate_split(
            specs,
            feature_store,
            y,
            random_train_df["_row_id"].to_numpy(dtype=int),
            random_test_df["_row_id"].to_numpy(dtype=int),
            "random",
        )
    )
    rows.extend(
        evaluate_split(
            specs,
            feature_store,
            y,
            scaffold_train_df["_row_id"].to_numpy(dtype=int),
            scaffold_test_df["_row_id"].to_numpy(dtype=int),
            "scaffold",
        )
    )

    results = pd.DataFrame(rows)
    pivot = results.pivot(index=["model_family", "feature_set"], columns="split", values=["mae", "rmse", "r2"])
    pivot.columns = ["_".join(column).strip() for column in pivot.columns.to_flat_index()]
    pivot = pivot.reset_index()
    pivot = pivot.sort_values(
        ["rmse_scaffold", "r2_scaffold", "rmse_random"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "model_family_benchmark.csv"
    json_path = reports_dir / "model_family_benchmark.json"

    pivot.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_label_from_path(data_path),
                "dataset_path": str(data_path),
                "descriptor_names": DESCRIPTOR_NAMES,
                "results": pivot.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Saved model-family benchmark: {csv_path}")
    print(f"[OK] Saved model-family benchmark JSON: {json_path}")
    print(pivot.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
