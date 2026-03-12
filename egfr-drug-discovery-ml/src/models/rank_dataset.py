from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.features.featurize_ecfp import ecfp_from_smiles
from src.utils.drug_scores import qed_score, molecular_weight, logp, tpsa, hbd, hba
from src.utils.sa_score import simple_sa_score
from src.utils.advanced_filters import pains_alert


def property_penalty(
    mw: float | None,
    lp: float | None,
    psa: float | None,
    num_hbd: int | None,
    num_hba: int | None,
) -> float:
    penalty = 0.0

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


def main() -> None:
    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"
    data_path = PROCESSED_DIR / "egfr_chembl_ic50_clean.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ensemble model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {data_path}\n"
            "Run: python -m src.data.fetch_chembl_egfr && python -m src.data.clean_egfr_ic50"
        )

    models = joblib.load(model_path)
    df = pd.read_csv(data_path)

    if "smiles_canonical" not in df.columns:
        raise ValueError(
            f"Expected column 'smiles_canonical'. Found: {list(df.columns)}"
        )

    smiles_list = df["smiles_canonical"].dropna().tolist()

    print("Computing fingerprints...")
    X = np.vstack([ecfp_from_smiles(s) for s in tqdm(smiles_list, desc="Fingerprints")])

    print("Running ensemble predictions...")
    preds = []
    for m in models:
        preds.append(m.predict(X))
    preds = np.vstack(preds)

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    print("Computing descriptors + filters...")
    rows = []

    for i, smiles in enumerate(tqdm(smiles_list, desc="Descriptors")):
        try:
            qed = qed_score(smiles)
            mw = molecular_weight(smiles)
            lp = logp(smiles)
            psa = tpsa(smiles)
            num_hbd = hbd(smiles)
            num_hba = hba(smiles)

            penalty = property_penalty(mw, lp, psa, num_hbd, num_hba)

            sa = simple_sa_score(smiles)
            if sa is None:
                sa = 10.0

            has_pains, pains_desc = pains_alert(smiles)
            pains_penalty = 0.5 if has_pains else 0.0

            if qed is None:
                qed = 0.0

            final_score = (
                float(pred_mean[i])
                + 0.5 * float(qed)
                - float(pred_std[i])
                - float(penalty)
                - 0.15 * float(sa)
                - float(pains_penalty)
            )

            rows.append({
                "smiles": smiles,
                "predicted_pIC50": float(pred_mean[i]),
                "uncertainty": float(pred_std[i]),
                "QED": float(qed),
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
            })

        except Exception as e:
            print(f"[WARN] Skipping molecule: {smiles} | {repr(e)}")

    if not rows:
        raise ValueError("No molecules were successfully scored.")

    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["smiles"])
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
    )

    out_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n[OK] Saved ranked dataset: {out_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()