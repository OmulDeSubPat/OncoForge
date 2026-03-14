from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.pipelines.artifact_utils import load_csv_artifact


def main():
    in_path = PROJECT_ROOT / "reports" / "candidates_vs_market.csv"
    market_path = PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing file: {in_path}\n"
            "Run: python -m src.benchmark.compare_candidates_to_market"
        )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Missing file: {market_path}\n"
            "Run: python -m src.benchmark.score_marketed_egfr"
        )

    df = load_csv_artifact(
        in_path,
        required_columns=["predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "audit_status", "veto", "max_market_similarity", "final_score"],
        producer="python -m src.benchmark.compare_candidates_to_market",
    )
    market = load_csv_artifact(
        market_path,
        required_columns=["predicted_pIC50", "final_score"],
        producer="python -m src.benchmark.score_marketed_egfr",
    )

    market_score_ref = market["final_score"].median()
    market_pic50_ref = market["predicted_pIC50"].median()

    shortlist = df[
        (df["predicted_pIC50"] >= market_pic50_ref - 0.3)
        & (df["final_score"] >= market_score_ref - 0.3)
        & (df["QED"] >= 0.40)
        & (df["reward_hacking_risk"] <= 0.35)
        & (df["agent_disagreement_score"] <= 0.50)
        & (df["audit_status"] == "pass")
        & (df["veto"] == False)
        & (df["max_market_similarity"] <= 0.85)
    ].copy()

    if "docking_rescore" in shortlist.columns:
        shortlist = shortlist[shortlist["docking_rescore"] >= 0.45].copy()
        shortlist["structural_priority_score"] = shortlist["final_score"] + 0.75 * shortlist["docking_rescore"]
        shortlist = shortlist.sort_values(
            ["structural_priority_score", "docking_rescore", "final_score", "predicted_pIC50", "max_market_similarity"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    else:
        shortlist = shortlist.sort_values(
            ["final_score", "predicted_pIC50", "max_market_similarity"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv"
    shortlist.to_csv(out_path, index=False)

    print(f"[OK] Saved shortlist: {out_path}")
    print(f"[INFO] Market median final_score = {market_score_ref:.3f}")
    print(f"[INFO] Market median predicted_pIC50 = {market_pic50_ref:.3f}")
    print(shortlist.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
