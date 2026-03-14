from __future__ import annotations

import pandas as pd

from src.agents.multi_agent import add_multiobjective_ranking, build_default_scorer
from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule


def main():
    in_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {in_path}")

    df = pd.read_csv(in_path)
    scorer = build_default_scorer()

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
            scored = score_molecule(smi, scorer=scorer)
            scored["name"] = name
            scored["class"] = klass
            rows.append(scored)
            print(f"[OK] Scored {name}")
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"[WARN] Skipping {name}: {repr(exc)}")

    if not rows:
        raise ValueError("No benchmark molecules were successfully scored.")

    out = add_multiobjective_ranking(pd.DataFrame(rows), policy=scorer.policy)

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
