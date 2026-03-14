from __future__ import annotations

import numpy as np


def summarize_uncertainty_calibration(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target_coverage: float = 0.6826894921370859,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std, dtype=float), 1e-6)
    abs_error = np.abs(y_true - pred_mean)

    raw_coverage = float(np.mean(abs_error <= pred_std))
    raw_two_sigma_coverage = float(np.mean(abs_error <= 2.0 * pred_std))

    scale_grid = np.linspace(0.25, 4.0, 151)
    coverages = np.asarray([np.mean(abs_error <= scale * pred_std) for scale in scale_grid], dtype=float)
    best_idx = int(np.argmin(np.abs(coverages - target_coverage)))
    scale_factor = float(scale_grid[best_idx])
    scaled_std = pred_std * scale_factor

    calibrated_coverage = float(np.mean(abs_error <= scaled_std))
    calibrated_two_sigma_coverage = float(np.mean(abs_error <= 2.0 * scaled_std))
    normalized_abs_error = abs_error / scaled_std

    return {
        "target_one_sigma_coverage": float(target_coverage),
        "raw_one_sigma_coverage": raw_coverage,
        "raw_two_sigma_coverage": raw_two_sigma_coverage,
        "recommended_uncertainty_scale": scale_factor,
        "calibrated_one_sigma_coverage": calibrated_coverage,
        "calibrated_two_sigma_coverage": calibrated_two_sigma_coverage,
        "mean_abs_error": float(abs_error.mean()),
        "mean_predicted_std": float(pred_std.mean()),
        "mean_calibrated_std": float(scaled_std.mean()),
        "mean_normalized_abs_error": float(normalized_abs_error.mean()),
    }
