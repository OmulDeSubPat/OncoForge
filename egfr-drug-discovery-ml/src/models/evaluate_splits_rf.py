from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR
from src.features.featurize_ecfp import ecfp_from_smiles
from src.evaluation.scaffold_split import scaffold_split
from src.evaluation.random_split import random_split


def featurize_df(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    return np.vstack([ecfp_from_smiles(s) for s in df[smiles_col].tolist()])


def train_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, smiles_col: str, y_col: str):
    X_train = featurize_df(train_df, smiles_col)
    y_train = train_df[y_col].values.astype(float)

    X_test = featurize_df(test_df, smiles_col)
    y_test = test_df[y_col].values.astype(float)

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    return mae, rmse, r2, len(train_df), len(test_df)


def main():
    df = pd.read_csv(PROCESSED_DIR / "egfr_chembl_ic50_clean.csv")

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"

    # Random split
    tr_r, te_r = random_split(df, test_size=0.2, seed=42)
    mae_r, rmse_r, r2_r, ntr_r, nte_r = train_eval(tr_r, te_r, smiles_col, y_col)

    # Scaffold split
    tr_s, te_s = scaffold_split(df, smiles_col=smiles_col, test_size=0.2, seed=42)
    mae_s, rmse_s, r2_s, ntr_s, nte_s = train_eval(tr_s, te_s, smiles_col, y_col)

    print("=== Random split (optimistic) ===")
    print(f"MAE:  {mae_r:.3f} | RMSE: {rmse_r:.3f} | R2: {r2_r:.3f} | train={ntr_r} test={nte_r}")

    print("\n=== Scaffold split (realistic) ===")
    print(f"MAE:  {mae_s:.3f} | RMSE: {rmse_s:.3f} | R2: {r2_s:.3f} | train={ntr_s} test={nte_s}")


if __name__ == "__main__":
    main()