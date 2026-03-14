from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.similarity import morgan_fp, tanimoto_similarity


def main():
    preferred_path = PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv"
    in_path = preferred_path if preferred_path.exists() else (PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv")
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing file: {in_path}\n"
            "Run: python -m src.generation.generate_and_rank_analogs"
        )

    df = load_csv_artifact(
        in_path,
        required_columns=["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "audit_status", "veto", "final_score"],
        producer="python -m src.generation.generate_and_rank_analogs",
    )
    df = df[
        (df["predicted_pIC50"] >= 8.3)
        & (df["QED"] >= 0.35)
        & (df["reward_hacking_risk"] <= 0.35)
        & (df["agent_disagreement_score"] <= 0.50)
        & (df["audit_status"] == "pass")
        & (df["veto"] == False)
    ].copy()

    if "docking_rescore" in df.columns:
        df = df[df["docking_rescore"] >= 0.45].copy()
        sort_cols = ["structural_priority_score", "docking_rescore", "final_score"]
    else:
        sort_cols = ["final_score"]
    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    selected = []
    selected_fps = []

    similarity_threshold = 0.72
    max_candidates = 20

    for _, row in df.iterrows():
        fp = morgan_fp(smiles=row["smiles"])
        if fp is None:
            continue

        too_similar = any(
            tanimoto_similarity(fp, prev_fp) >= similarity_threshold
            for prev_fp in selected_fps
        )
        if too_similar:
            continue

        selected.append(row.to_dict())
        selected_fps.append(fp)

        if len(selected) >= max_candidates:
            break

    out = pd.DataFrame(selected)

    out_path = PROJECT_ROOT / "reports" / "final_diverse_candidates.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved final diverse candidates: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
