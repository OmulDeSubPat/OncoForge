from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.evaluation.scaffold_split import scaffold_split


def featurize_df(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    return np.vstack([ecfp_from_smiles(s) for s in df[smiles_col].tolist()])


def main() -> None:
    data_path = PROCESSED_DIR / "egfr_chembl_ic50_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {data_path}\n"
            "Run: python -m src.data.fetch_chembl_egfr && python -m src.data.clean_egfr_ic50"
        )

    df = pd.read_csv(data_path)

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"

    if smiles_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns '{smiles_col}' and '{y_col}'. Found: {list(df.columns)}")

    # Scaffold split for realistic generalization
    train_df, test_df = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=42)

    X_train = featurize_df(train_df, smiles_col)
    y_train = train_df[y_col].values.astype(float)

    X_test = featurize_df(test_df, smiles_col)
    y_test = test_df[y_col].values.astype(float)

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    print("[RESULT] Scaffold split evaluation:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qsar_rf_scaffold.pkl"
    joblib.dump(model, out_path)

    print(f"[OK] Model saved: {out_path}")


if __name__ == "__main__":
    main()
