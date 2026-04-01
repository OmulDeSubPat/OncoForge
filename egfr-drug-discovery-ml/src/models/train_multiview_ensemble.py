from __future__ import annotations

import json
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT
from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.evaluation.calibration import summarize_uncertainty_calibration
from src.evaluation.random_split import random_split
from src.evaluation.scaffold_split import scaffold_split
from src.evaluation.temporal_split import temporal_split
from src.features.descriptor_features import DESCRIPTOR_NAMES, descriptor_vector_from_smiles
from src.features.featurize_ecfp import ecfp_from_smiles
from src.pipelines.reproducibility import write_metric_snapshot


def metrics_dict(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def build_features(df: pd.DataFrame, smiles_col: str) -> dict[str, np.ndarray]:
    ecfp = np.vstack([ecfp_from_smiles(smiles) for smiles in df[smiles_col].tolist()]).astype(np.float32)
    descriptors = np.vstack([descriptor_vector_from_smiles(smiles) for smiles in df[smiles_col].tolist()]).astype(np.float32)
    hybrid = np.hstack([ecfp, descriptors]).astype(np.float32)
    return {
        "ecfp": ecfp,
        "descriptors": descriptors,
        "hybrid": hybrid,
    }


def _select_temporal_split_config(df: pd.DataFrame) -> dict[str, object]:
    year_col = "temporal_year_max" if "temporal_year_max" in df.columns else "year_max"
    year_min_col = "temporal_year_min" if "temporal_year_min" in df.columns else "year_min"
    config: dict[str, object] = {
        "year_col": year_col,
        "year_min_col": year_min_col,
        "test_size": 0.2,
    }
    if "source_dataset" in df.columns:
        config["source_col"] = "source_dataset"
    elif "source_datasets" in df.columns:
        config["source_col"] = "source_datasets"
    return config


def _snapshot_payload(metrics: dict[str, object] | None) -> dict[str, float]:
    if not metrics:
        return {}
    payload: dict[str, float] = {}
    for metric_name, source_key in {
        "R2": "r2",
        "RMSE": "rmse",
        "MAE": "mae",
        "Incertitudine": "avg_uncertainty",
    }.items():
        value = metrics.get(source_key)
        if value is None:
            continue
        payload[metric_name] = float(value)
    return payload


def _write_split_snapshot(
    *,
    split_name: str,
    metrics: dict[str, object] | None,
    dataset_name: str,
    observations: str,
) -> None:
    snapshot = _snapshot_payload(metrics)
    if not snapshot:
        return
    write_metric_snapshot(
        metrics=snapshot,
        version="multiview_ensemble",
        experiment_name=f"train_multiview_ensemble_{dataset_name}",
        split=split_name,
        observations=observations,
    )


def make_model_bundles():
    return [
        {
            "name": "et_ecfp",
            "feature_set": "ecfp",
            "model": ExtraTreesRegressor(
                n_estimators=700,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
            ),
        },
        {
            "name": "hgb_ecfp",
            "feature_set": "ecfp",
            "model": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.05,
                early_stopping=False,
                random_state=42,
            ),
        },
        {
            "name": "rf_hybrid",
            "feature_set": "hybrid",
            "model": RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                max_features=0.18,
                bootstrap=True,
                max_samples=0.85,
            ),
        },
    ]


def _predict_bundle(bundle: dict, feature_store: dict[str, np.ndarray], row_idx: np.ndarray) -> np.ndarray:
    x = feature_store[bundle["feature_set"]][row_idx]
    return np.asarray(bundle["model"].predict(x), dtype=float)


def evaluate_ensemble(
    bundles_template: list[dict],
    feature_store: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, object]:
    trained_bundles = []
    preds = []
    for bundle in bundles_template:
        fitted = {
            "name": bundle["name"],
            "feature_set": bundle["feature_set"],
            "model": bundle["model"],
        }
        x_train = feature_store[bundle["feature_set"]][train_idx]
        fitted["model"].fit(x_train, y[train_idx])
        pred = _predict_bundle(fitted, feature_store, test_idx)
        preds.append(pred)
        trained_bundles.append(fitted)

    pred_matrix = np.vstack(preds).astype(float)
    pred_mean = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)
    metrics = metrics_dict(y[test_idx], pred_mean)
    metrics["avg_uncertainty"] = float(pred_std.mean())
    metrics["n_models"] = int(len(trained_bundles))
    metrics["pred_mean"] = pred_mean
    metrics["pred_std"] = pred_std
    return metrics


def main() -> None:
    data_path = resolve_preferred_processed_dataset()
    dataset_name = dataset_label_from_path(data_path)
    df = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"
    y = df[y_col].to_numpy(dtype=float)

    print("[INFO] Building multiview feature store...")
    feature_store = build_features(df, smiles_col)

    random_train_df, random_test_df = random_split(df, test_size=0.2, seed=42)
    scaffold_train_df, scaffold_test_df = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=42)

    bundles_for_random = make_model_bundles()
    bundles_for_scaffold = make_model_bundles()

    random_metrics = evaluate_ensemble(
        bundles_for_random,
        feature_store,
        y,
        random_train_df["_row_id"].to_numpy(dtype=int),
        random_test_df["_row_id"].to_numpy(dtype=int),
    )
    scaffold_metrics = evaluate_ensemble(
        bundles_for_scaffold,
        feature_store,
        y,
        scaffold_train_df["_row_id"].to_numpy(dtype=int),
        scaffold_test_df["_row_id"].to_numpy(dtype=int),
    )
    random_calibration = summarize_uncertainty_calibration(
        y[random_test_df["_row_id"].to_numpy(dtype=int)],
        random_metrics["pred_mean"],
        random_metrics["pred_std"],
    )
    scaffold_calibration = summarize_uncertainty_calibration(
        y[scaffold_test_df["_row_id"].to_numpy(dtype=int)],
        scaffold_metrics["pred_mean"],
        scaffold_metrics["pred_std"],
    )

    temporal_metrics: dict[str, object] | None = None
    temporal_metadata: dict[str, object] | None = None
    temporal_calibration: dict[str, float] | None = None
    try:
        temporal_split_kwargs = _select_temporal_split_config(df)
        print("[INFO] Temporal split config:", temporal_split_kwargs)
        temporal_train_df, temporal_test_df, temporal_metadata = temporal_split(df, **temporal_split_kwargs)
        bundles_for_temporal = make_model_bundles()
        temporal_metrics = evaluate_ensemble(
            bundles_for_temporal,
            feature_store,
            y,
            temporal_train_df["_row_id"].to_numpy(dtype=int),
            temporal_test_df["_row_id"].to_numpy(dtype=int),
        )
        temporal_calibration = summarize_uncertainty_calibration(
            y[temporal_test_df["_row_id"].to_numpy(dtype=int)],
            temporal_metrics["pred_mean"],
            temporal_metrics["pred_std"],
        )
        print("[INFO] Temporal split multiview ensemble:", {k: v for k, v in temporal_metrics.items() if k not in {"pred_mean", "pred_std"}})
        print("[INFO] Temporal split metadata:", temporal_metadata)
        if temporal_metadata is not None:
            print("[INFO] Temporal train source counts:", temporal_metadata.get("train_source_counts", {}))
            print("[INFO] Temporal test source counts:", temporal_metadata.get("test_source_counts", {}))
    except ValueError as exc:
        print(f"[WARN] Temporal split skipped: {exc}")

    print("[INFO] Random split multiview ensemble:", {k: v for k, v in random_metrics.items() if k not in {"pred_mean", "pred_std"}})
    print("[INFO] Scaffold split multiview ensemble:", {k: v for k, v in scaffold_metrics.items() if k not in {"pred_mean", "pred_std"}})

    print("[INFO] Training final multiview ensemble on full dataset...")
    final_bundles = make_model_bundles()
    for bundle in final_bundles:
        bundle["model"].fit(feature_store[bundle["feature_set"]], y)

    uncertainty_scale = float(
        max(
            random_calibration["recommended_uncertainty_scale"],
            scaffold_calibration["recommended_uncertainty_scale"],
        )
    )
    if temporal_calibration is not None:
        uncertainty_scale = float(max(uncertainty_scale, temporal_calibration["recommended_uncertainty_scale"]))

    model_payload = {
        "models": [
            {"name": bundle["name"], "feature_set": bundle["feature_set"], "model": bundle["model"]}
            for bundle in final_bundles
        ],
        "uncertainty_scale": uncertainty_scale,
        "dataset_name": dataset_name,
        "dataset_path": str(data_path),
        "descriptor_names": DESCRIPTOR_NAMES,
    }

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "qsar_multiview_ensemble.pkl"
    metadata_path = models_dir / "qsar_multiview_ensemble_metadata.json"
    joblib.dump(model_payload, model_path, compress=3)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_label_from_path(data_path),
                "dataset_name": dataset_name,
                "dataset_path": str(data_path),
                "descriptor_names": DESCRIPTOR_NAMES,
                "models": [
                    {
                        "name": bundle["name"],
                        "feature_set": bundle["feature_set"],
                        "model_class": bundle["model"].__class__.__name__,
                    }
                    for bundle in final_bundles
                ],
                "recommended_uncertainty_scale": uncertainty_scale,
                "random_calibration": random_calibration,
                "scaffold_calibration": scaffold_calibration,
                "temporal_calibration": temporal_calibration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "model_performance_summary.json"
    metrics = {
        "dataset_size": int(len(df)),
        "dataset_name": dataset_name,
        "dataset_path": str(data_path),
        "smiles_column": smiles_col,
        "target_column": y_col,
        "model_type": "multiview_ensemble",
        "n_models": len(final_bundles),
        "selected_models": [bundle["name"] for bundle in final_bundles],
        "feature_sets": {bundle["name"]: bundle["feature_set"] for bundle in final_bundles},
        "random_split": {
            **{k: v for k, v in random_metrics.items() if k not in {"pred_mean", "pred_std"}},
            "train_size": int(len(random_train_df)),
            "test_size": int(len(random_test_df)),
        },
        "scaffold_split": {
            **{k: v for k, v in scaffold_metrics.items() if k not in {"pred_mean", "pred_std"}},
            "train_size": int(len(scaffold_train_df)),
            "test_size": int(len(scaffold_test_df)),
        },
        "uncertainty_calibration": {
            "recommended_uncertainty_scale": uncertainty_scale,
            "random_split": random_calibration,
            "scaffold_split": scaffold_calibration,
            "temporal_split": temporal_calibration,
        },
        "temporal_split_config": _select_temporal_split_config(df),
    }
    if temporal_metrics is not None and temporal_metadata is not None:
        metrics["temporal_split"] = {
            **{k: v for k, v in temporal_metrics.items() if k not in {"pred_mean", "pred_std"}},
            **temporal_metadata,
        }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write_split_snapshot(
        split_name="random",
        metrics=random_metrics,
        dataset_name=dataset_name,
        observations=(
            f"dataset={dataset_name}; train_size={len(random_train_df)}; "
            f"test_size={len(random_test_df)}"
        ),
    )
    _write_split_snapshot(
        split_name="scaffold",
        metrics=scaffold_metrics,
        dataset_name=dataset_name,
        observations=(
            f"dataset={dataset_name}; train_size={len(scaffold_train_df)}; "
            f"test_size={len(scaffold_test_df)}"
        ),
    )
    if temporal_metrics is not None:
        temporal_observations = [
            f"dataset={dataset_name}",
            f"strategy={(temporal_metadata or {}).get('strategy', 'n/a')}",
            f"cutoff_year={(temporal_metadata or {}).get('cutoff_year', 'n/a')}",
            f"train_size={(temporal_metadata or {}).get('train_size', 'n/a')}",
            f"test_size={(temporal_metadata or {}).get('test_size', 'n/a')}",
            f"source_imbalance={(temporal_metadata or {}).get('source_imbalance', 'n/a')}",
        ]
        _write_split_snapshot(
            split_name="temporal",
            metrics=temporal_metrics,
            dataset_name=dataset_name,
            observations="; ".join(temporal_observations),
        )

    print(f"[OK] Saved multiview model: {model_path}")
    print(f"[OK] Saved multiview metadata: {metadata_path}")
    print(f"[OK] Saved performance summary: {metrics_path}")


if __name__ == "__main__":
    main()
