from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT


TOP_K_VALUES = [25, 50, 100, 250]


def _evaluate_top_k(df: pd.DataFrame, strategy_name: str, score_column: str, top_k: int) -> dict[str, object]:
    ranked = df.sort_values(score_column, ascending=False).head(top_k).copy()
    return {
        "strategy": strategy_name,
        "score_column": score_column,
        "top_k": int(top_k),
        "mean_predicted_pIC50": float(ranked["predicted_pIC50"].mean()),
        "mean_qed": float(ranked["QED"].mean()),
        "mean_sa_score": float(ranked["SA_score"].mean()),
        "mean_reward_hacking_risk": float(ranked["reward_hacking_risk"].mean()),
        "mean_applicability": float(ranked["applicability_score"].mean()),
        "mean_uncertainty": float(ranked["uncertainty"].mean()),
        "audit_pass_rate": float((ranked["audit_pass"] == True).mean()),
        "veto_rate": float((ranked["veto"] == True).mean()),
        "review_or_fail_rate": float(ranked["audit_status"].isin(["review", "fail"]).mean()),
        "mean_max_market_similarity": float(ranked["max_market_similarity"].mean()),
    }


def main() -> None:
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    df = pd.read_csv(ranked_path, low_memory=False)

    df["verified_plus_mo"] = df["verified_reward"] + 1.20 * df["multi_objective_score"]
    df["no_risk_penalty_score"] = (
        df["verified_reward"]
        + 1.20 * df["multi_objective_score"]
        - 2.50 * df["veto"].astype(float)
        - df["audit_status_penalty"]
    )
    df["no_audit_status_score"] = (
        df["verified_reward"]
        + 1.20 * df["multi_objective_score"]
        - 1.50 * df["reward_hacking_risk"]
        - 2.50 * df["veto"].astype(float)
    )
    df["no_veto_score"] = (
        df["verified_reward"]
        + 1.20 * df["multi_objective_score"]
        - 1.50 * df["reward_hacking_risk"]
        - df["audit_status_penalty"]
    )

    strategies = [
        ("naive_proxy", "naive_score"),
        ("verified_plus_mo", "verified_plus_mo"),
        ("protected_final", "final_score"),
        ("no_risk_penalty", "no_risk_penalty_score"),
        ("no_audit_status", "no_audit_status_score"),
        ("no_veto", "no_veto_score"),
    ]

    rows = []
    for strategy_name, score_column in strategies:
        for top_k in TOP_K_VALUES:
            rows.append(_evaluate_top_k(df, strategy_name, score_column, top_k))

    out_df = pd.DataFrame(rows).sort_values(["top_k", "mean_reward_hacking_risk", "mean_predicted_pIC50"]).reset_index(drop=True)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "multi_agent_ablation.csv"
    json_path = reports_dir / "multi_agent_ablation.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(out_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Saved multi-agent ablation CSV: {csv_path}")
    print(f"[OK] Saved multi-agent ablation JSON: {json_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
