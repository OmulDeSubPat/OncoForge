from __future__ import annotations

import joblib
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule


def main():
    in_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
    model_path = PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {in_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )

    df = pd.read_csv(in_path)
    models = joblib.load(model_path)

    if "smiles" not in df.columns:
        raise ValueError("Benchmark CSV must contain a 'smiles' column.")

    rows = []
    failed = []

    for _, row in df.iterrows():
        smi = row.get("smiles", "")
        name = row.get("name", "unknown")
        klass = row.get("class", "unknown")

        if not isinstance(smi, str) or not smi.strip():
            failed.append((name, "empty_smiles"))
            continue

        try:
            scored = score_molecule(smi, models)
            scored["name"] = name
            scored["class"] = klass
            rows.append(scored)
            print(f"[OK] Scored {name}")
        except Exception as e:
            failed.append((name, repr(e)))
            print(f"[WARN] Skipping {name}: {repr(e)}")

    if not rows:
        raise ValueError("No benchmark molecules were successfully scored.")

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv"
    out.to_csv(out_path, index=False)

    print(f"\n[OK] Saved scored marketed benchmark: {out_path}")
    print(out.to_string(index=False))

    if failed:
        failed_df = pd.DataFrame(failed, columns=["name", "error"])
        failed_path = PROJECT_ROOT / "reports" / "marketed_egfr_failed.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"\n[WARN] Some molecules failed and were skipped: {failed_path}")
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()