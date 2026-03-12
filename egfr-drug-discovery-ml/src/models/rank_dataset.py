from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import Chem

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.utils.drug_scores import qed_score, molecular_weight, logp, tpsa, hbd, hba


def main():

    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"
    models = joblib.load(model_path)

    df = pd.read_csv(PROCESSED_DIR / "egfr_chembl_ic50_clean.csv")

    smiles_list = df["smiles_canonical"].tolist()

    print("Computing fingerprints...")

    X = np.vstack([ecfp_from_smiles(s) for s in tqdm(smiles_list)])

    print("Running ensemble predictions...")

    preds = []
    for m in models:
        preds.append(m.predict(X))

    preds = np.vstack(preds)

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    print("Computing molecular descriptors...")

    rows = []

    for i, smiles in enumerate(tqdm(smiles_list)):

        qed = qed_score(smiles)
        mw = molecular_weight(smiles)
        lp = logp(smiles)
        psa = tpsa(smiles)
        num_hbd = hbd(smiles)
        num_hba = hba(smiles)

        penalty = 0

        if mw and mw > 500:
            penalty += 0.5
        if lp and lp > 5:
            penalty += 0.5
        if psa and psa > 140:
            penalty += 0.3
        if num_hbd and num_hbd > 5:
            penalty += 0.3
        if num_hba and num_hba > 10:
            penalty += 0.3

        final_score = pred_mean[i] + 0.5 * (qed or 0) - pred_std[i] - penalty

        rows.append({
            "smiles": smiles,
            "predicted_pIC50": pred_mean[i],
            "uncertainty": pred_std[i],
            "QED": qed,
            "MW": mw,
            "LogP": lp,
            "TPSA": psa,
            "HBD": num_hbd,
            "HBA": num_hba,
            "penalty": penalty,
            "final_score": final_score
        })

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, index=False)

    print(f"\n[OK] Saved ranked dataset: {out_path}")
    print(out.head(20))


if __name__ == "__main__":
    main()