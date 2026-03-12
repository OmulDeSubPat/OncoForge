from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.utils.drug_scores import (
    qed_score,
    molecular_weight,
    logp,
    tpsa,
    hbd,
    hba,
)
from src.utils.sa_score import simple_sa_score
from src.utils.advanced_filters import pains_alert


def load_ensemble():
    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ensemble model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )
    return joblib.load(model_path)


def predict_with_ensemble(smiles: str, models) -> tuple[float, float]:
    x = ecfp_from_smiles(smiles).reshape(1, -1)
    preds = np.array([m.predict(x)[0] for m in models], dtype=float)
    return float(preds.mean()), float(preds.std())


def property_penalty(smiles: str) -> float:
    penalty = 0.0

    mw = molecular_weight(smiles)
    lp = logp(smiles)
    psa = tpsa(smiles)
    num_hbd = hbd(smiles)
    num_hba = hba(smiles)

    if mw is None or lp is None or psa is None or num_hbd is None or num_hba is None:
        return 2.0

    if mw > 500:
        penalty += 0.5
    if lp > 5:
        penalty += 0.5
    if psa > 140:
        penalty += 0.3
    if num_hbd > 5:
        penalty += 0.3
    if num_hba > 10:
        penalty += 0.3

    return penalty


def score_molecule(smiles: str, models) -> dict:
    pred_mean, pred_std = predict_with_ensemble(smiles, models)

    qed = qed_score(smiles)
    mw = molecular_weight(smiles)
    lp = logp(smiles)
    psa = tpsa(smiles)
    num_hbd = hbd(smiles)
    num_hba = hba(smiles)

    penalty = property_penalty(smiles)

    sa = simple_sa_score(smiles)
    if sa is None:
        sa = 10.0

    has_pains, pains_desc = pains_alert(smiles)
    pains_penalty = 0.5 if has_pains else 0.0

    if qed is None:
        qed = 0.0

    # scor nou
    final_score = (
        pred_mean
        + 0.5 * qed
        - pred_std
        - penalty
        - 0.15 * sa
        - pains_penalty
    )

    return {
        "smiles": smiles,
        "predicted_pIC50": pred_mean,
        "uncertainty": pred_std,
        "QED": qed,
        "MW": mw,
        "LogP": lp,
        "TPSA": psa,
        "HBD": num_hbd,
        "HBA": num_hba,
        "penalty": penalty,
        "SA_score": sa,
        "has_PAINS": has_pains,
        "PAINS_alert": pains_desc,
        "PAINS_penalty": pains_penalty,
        "final_score": final_score,
    }


def main():
    models = load_ensemble()

    sample_smiles = [
        "CCO",
        "c1ccccc1",
        "CN(C)CCOC(c1ccccc1)c1ccccc1",
    ]

    rows = [score_molecule(s, models) for s in sample_smiles]
    df = pd.DataFrame(rows)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()