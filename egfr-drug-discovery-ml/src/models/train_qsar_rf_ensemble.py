from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.evaluation.scaffold_split import scaffold_split


def featurize_df(df, smiles_col):
    return np.vstack([ecfp_from_smiles(s) for s in df[smiles_col].tolist()])


def main():

    df = pd.read_csv(PROCESSED_DIR / "egfr_chembl_ic50_clean.csv")

    smiles_col = "smiles_canonical"
    y_col = "pIC50_median"

    train_df, test_df = scaffold_split(df, smiles_col=smiles_col)

    X_train = featurize_df(train_df, smiles_col)
    y_train = train_df[y_col].values

    X_test = featurize_df(test_df, smiles_col)
    y_test = test_df[y_col].values

    seeds = [1,2,3,4,5]

    models = []
    preds = []

    for s in seeds:

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=s,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        p = model.predict(X_test)

        models.append(model)
        preds.append(p)

    preds = np.vstack(preds)

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    mae = mean_absolute_error(y_test, pred_mean)
    rmse = mean_squared_error(y_test, pred_mean)**0.5
    r2 = r2_score(y_test, pred_mean)

    print("Ensemble results")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    print("Average uncertainty:", pred_std.mean())

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True)

    joblib.dump(models, out_dir / "qsar_rf_ensemble.pkl")

    print("Saved ensemble model")


if __name__ == "__main__":
    main()