from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.models.train_multiview_ensemble import build_features, evaluate_ensemble, metrics_dict, make_model_bundles


def _exclusive_source_series(df: pd.DataFrame) -> pd.Series:
    source_series = df.get("source_datasets", pd.Series("", index=df.index)).fillna("").astype(str)
    n_sources = pd.to_numeric(df.get("n_sources", 1), errors="coerce").fillna(1).astype(int)
    exclusive = source_series.where(n_sources.eq(1), other=pd.NA)
    exclusive = exclusive.where(~exclusive.fillna("").str.contains(";"), other=pd.NA)
    return exclusive


def _strong_recall(y_true: np.ndarray, pred: np.ndarray, fraction: float, threshold: float) -> tuple[float, float]:
    positive_mask = y_true >= threshold
    positive_count = int(positive_mask.sum())
    if positive_count == 0:
        return 0.0, 0.0
    top_k = max(1, int(np.ceil(len(y_true) * float(fraction))))
    order = np.argsort(-pred)
    top_positive_mask = positive_mask[order[:top_k]]
    recall = float(top_positive_mask.sum() / positive_count)
    precision = float(top_positive_mask.mean()) if top_positive_mask.size else 0.0
    return recall, precision


def _median_positive_rank(y_true: np.ndarray, pred: np.ndarray, threshold: float) -> float | None:
    positive_mask = y_true >= threshold
    if not positive_mask.any():
        return None
    order = np.argsort(-pred)
    inverse_rank = np.empty_like(order)
    inverse_rank[order] = np.arange(1, len(order) + 1)
    positive_ranks = inverse_rank[positive_mask]
    return float(np.median(positive_ranks))


def _plot_holdout_rmse(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values("rmse").copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(plot_df["source"], plot_df["rmse"], color="#457b9d")
    ax.set_ylabel("RMSE")
    ax.set_title("Leave-One-Source-Out Generalization")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, plot_df["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_holdout_recall(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values("recall_top20pct", ascending=False).copy()
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - width / 2, plot_df["recall_top10pct"], width, label="Recall @ top 10%", color="#2a9d8f")
    ax.bar(x + width / 2, plot_df["recall_top20pct"], width, label="Recall @ top 20%", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["source"], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Strong-active recall")
    ax.set_title("Source Holdout Recovery of Strong EGFR Actives")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark source holdout generalization on the multisource EGFR dataset.")
    parser.add_argument("--min-test-size", type=int, default=120, help="Minimum number of exclusive molecules required for a source holdout.")
    parser.add_argument("--strong-threshold", type=float, default=8.5, help="pIC50 threshold used to define strong actives.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional list of exclusive source labels to benchmark. Defaults to all eligible sources.",
    )
    args = parser.parse_args(argv)

    data_path = resolve_preferred_processed_dataset()
    df = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    if "smiles_canonical" not in df.columns or "pIC50_median" not in df.columns or "source_datasets" not in df.columns:
        raise ValueError("Source holdout benchmark requires smiles_canonical, pIC50_median, and source_datasets columns.")

    df["_row_id"] = np.arange(len(df))
    df["_exclusive_source"] = _exclusive_source_series(df)
    eligible_counts = df["_exclusive_source"].dropna().value_counts()
    eligible_sources = eligible_counts[eligible_counts >= int(args.min_test_size)].index.tolist()
    if args.sources:
        requested = [str(source) for source in args.sources]
        eligible_sources = [source for source in eligible_sources if source in requested]
    if not eligible_sources:
        raise ValueError("No exclusive sources met the requested source-holdout criteria.")

    y = df["pIC50_median"].to_numpy(dtype=float)
    feature_store = build_features(df, "smiles_canonical")

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for source in eligible_sources:
        test_mask = df["_exclusive_source"].eq(source)
        train_mask = ~test_mask
        train_idx = df.loc[train_mask, "_row_id"].to_numpy(dtype=int)
        test_idx = df.loc[test_mask, "_row_id"].to_numpy(dtype=int)
        if len(test_idx) < int(args.min_test_size) or len(train_idx) < 500:
            continue

        metrics = evaluate_ensemble(make_model_bundles(), feature_store, y, train_idx, test_idx)
        pred_mean = np.asarray(metrics["pred_mean"], dtype=float)
        pred_std = np.asarray(metrics["pred_std"], dtype=float)
        y_test = y[test_idx]
        baseline_pred = np.full_like(y_test, fill_value=float(y[train_idx].mean()), dtype=float)
        baseline_metrics = metrics_dict(y_test, baseline_pred)
        recall_top10, precision_top10 = _strong_recall(y_test, pred_mean, fraction=0.10, threshold=float(args.strong_threshold))
        recall_top20, precision_top20 = _strong_recall(y_test, pred_mean, fraction=0.20, threshold=float(args.strong_threshold))
        median_pos_rank = _median_positive_rank(y_test, pred_mean, threshold=float(args.strong_threshold))

        pred_df = df.loc[test_mask, ["smiles_canonical", "pIC50_median", "source_datasets"]].copy()
        pred_df["source"] = source
        pred_df["predicted_pIC50"] = pred_mean
        pred_df["predicted_uncertainty"] = pred_std
        pred_df["absolute_error"] = np.abs(pred_df["pIC50_median"] - pred_df["predicted_pIC50"])
        pred_df["is_strong_active"] = pred_df["pIC50_median"] >= float(args.strong_threshold)
        pred_df = pred_df.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)
        pred_df["predicted_rank"] = np.arange(1, len(pred_df) + 1)
        prediction_rows.append(pred_df)

        summary_rows.append(
            {
                "source": source,
                "test_size": int(len(test_idx)),
                "train_size": int(len(train_idx)),
                "n_strong_actives": int((y_test >= float(args.strong_threshold)).sum()),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                "avg_uncertainty": float(metrics["avg_uncertainty"]),
                "baseline_rmse": float(baseline_metrics["rmse"]),
                "rmse_gain_vs_baseline": float(baseline_metrics["rmse"] - metrics["rmse"]),
                "recall_top10pct": recall_top10,
                "precision_top10pct": precision_top10,
                "recall_top20pct": recall_top20,
                "precision_top20pct": precision_top20,
                "median_positive_rank": median_pos_rank,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["rmse", "recall_top20pct"], ascending=[True, False]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = reports_dir / "source_holdout_benchmark.csv"
    predictions_csv = reports_dir / "source_holdout_predictions.csv"
    summary_json = reports_dir / "source_holdout_benchmark.json"
    rmse_plot = reports_dir / "source_holdout_rmse.png"
    recall_plot = reports_dir / "source_holdout_recall.png"

    summary_df.to_csv(summary_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "dataset_name": dataset_label_from_path(data_path),
                "dataset_path": str(data_path),
                "min_test_size": int(args.min_test_size),
                "strong_threshold": float(args.strong_threshold),
                "results": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_holdout_rmse(summary_df, rmse_plot)
    _plot_holdout_recall(summary_df, recall_plot)

    print(f"[OK] Saved source holdout benchmark: {summary_csv}")
    print(f"[OK] Saved source holdout predictions: {predictions_csv}")
    print(f"[OK] Saved source holdout summary JSON: {summary_json}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
