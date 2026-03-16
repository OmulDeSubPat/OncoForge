from __future__ import annotations

import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.multi_agent import add_structure_agent_ranking
from src.config import PROJECT_ROOT
from src.feasibility.experimental_readiness import add_experimental_readiness
from src.pipelines.artifact_utils import load_csv_artifact


def main():
    in_path = PROJECT_ROOT / "reports" / "candidates_vs_market.csv"
    preferred_market_paths = [
        PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv",
        PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv",
    ]
    market_path = next((path for path in preferred_market_paths if path.exists()), preferred_market_paths[-1])

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
        producer="python -m src.structure.dock_marketed_egfr",
    )

    market_score_ref = market["final_score"].median()
    market_pic50_ref = market["predicted_pIC50"].median()
    market_docking_ref = (
        float(market["docking_rescore"].quantile(0.25))
        if "docking_rescore" in market.columns and not market["docking_rescore"].dropna().empty
        else None
    )
    market_interaction_ref = (
        float(market["interaction_support_score"].quantile(0.25))
        if "interaction_support_score" in market.columns and not market["interaction_support_score"].dropna().empty
        else None
    )

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
    if "feasibility_status" in shortlist.columns:
        shortlist = shortlist[(shortlist["feasibility_status"] == "pass") & (shortlist["feasibility_score"] >= 0.60)].copy()
    if "experimental_readiness_status" in shortlist.columns:
        shortlist = shortlist[shortlist["experimental_readiness_status"].isin(["ready", "supporting"])].copy()
    if "cross_database_status" in shortlist.columns:
        shortlist = shortlist[shortlist["cross_database_status"] != "weak"].copy()
    if "external_evidence_status" in shortlist.columns:
        shortlist = shortlist[shortlist["external_evidence_status"] != "fail"].copy()
    if "docking_rescore" in shortlist.columns:
        shortlist = shortlist[shortlist["docking_rescore"] >= 0.45].copy()
    if market_docking_ref is not None and "docking_rescore" in shortlist.columns:
        shortlist = shortlist[shortlist["docking_rescore"] >= (market_docking_ref - 0.05)].copy()
    if market_interaction_ref is not None and "interaction_support_score" in shortlist.columns:
        shortlist = shortlist[shortlist["interaction_support_score"] >= (market_interaction_ref - 0.08)].copy()

    if not shortlist.empty:
        shortlist = add_experimental_readiness(shortlist, market_df=market)
        shortlist = add_structure_agent_ranking(shortlist)
        shortlist = add_evidence_arbiter_ranking(shortlist)

    if "evidence_arbiter_priority" in shortlist.columns:
        shortlist = shortlist[shortlist["evidence_arbiter_status"] != "fail"].copy()
        sort_cols = [
            "evidence_arbiter_state_priority",
            "evidence_arbiter_priority",
            "experimental_readiness_priority" if "experimental_readiness_priority" in shortlist.columns else "structure_augmented_score",
            "structure_augmented_score" if "structure_augmented_score" in shortlist.columns else "predicted_pIC50",
            "predicted_pIC50",
            "max_market_similarity",
        ]
        ascending = [True, False, False, False, False, True]
        shortlist = shortlist.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "experimental_readiness_priority" in shortlist.columns:
        sort_cols = ["experimental_readiness_priority", "structure_augmented_score", "predicted_pIC50", "max_market_similarity"]
        ascending = [False, False, False, True]
        if "external_evidence_priority" in shortlist.columns:
            sort_cols = ["experimental_readiness_priority", "external_evidence_priority", "structure_augmented_score", "predicted_pIC50", "max_market_similarity"]
            ascending = [False, False, False, False, True]
        if "cross_database_consensus_score" in shortlist.columns:
            sort_cols = [
                "experimental_readiness_priority",
                "external_evidence_priority" if "external_evidence_priority" in shortlist.columns else "cross_database_consensus_score",
                "cross_database_consensus_score",
                "structure_augmented_score",
                "predicted_pIC50",
                "max_market_similarity",
            ]
            ascending = [False, False, False, False, False, True]
        shortlist = shortlist.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "feasible_priority_score" in shortlist.columns:
        sort_cols = ["feasible_priority_score", "final_score", "predicted_pIC50", "max_market_similarity"]
        ascending = [False, False, False, True]
        if "docking_rescore" in shortlist.columns:
            sort_cols = ["feasible_priority_score", "docking_rescore", "final_score", "predicted_pIC50", "max_market_similarity"]
            ascending = [False, False, False, False, True]
        shortlist = shortlist.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "docking_rescore" in shortlist.columns:
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

    if shortlist.empty:
        fallback = df[
            (df["predicted_pIC50"] >= market_pic50_ref - 0.45)
            & (df["final_score"] >= market_score_ref - 0.45)
            & (df["QED"] >= 0.38)
            & (df["reward_hacking_risk"] <= 0.35)
            & (df["agent_disagreement_score"] <= 0.55)
            & (df["audit_status"] != "fail")
            & (df["veto"] == False)
            & (df["max_market_similarity"] <= 0.88)
        ].copy()
        if "feasibility_status" in fallback.columns:
            fallback = fallback[(fallback["feasibility_status"] == "pass") & (fallback["feasibility_score"] >= 0.58)].copy()
        if "experimental_readiness_status" in fallback.columns:
            fallback = fallback[fallback["experimental_readiness_status"].isin(["ready", "supporting"])].copy()
        if "cross_database_status" in fallback.columns:
            fallback = fallback[fallback["cross_database_status"] != "weak"].copy()
        if "external_evidence_status" in fallback.columns:
            fallback = fallback[fallback["external_evidence_status"] != "fail"].copy()
        if "docking_rescore" in fallback.columns:
            fallback = fallback[fallback["docking_rescore"] >= 0.55].copy()
        if "interaction_support_score" in fallback.columns:
            fallback = fallback[fallback["interaction_support_score"] >= 0.65].copy()
        if not fallback.empty:
            fallback = add_experimental_readiness(fallback, market_df=market)
            fallback = add_structure_agent_ranking(fallback)
            fallback = add_evidence_arbiter_ranking(fallback)
        sort_cols = ["final_score", "predicted_pIC50", "max_market_similarity"]
        ascending = [False, False, True]
        if "evidence_arbiter_priority" in fallback.columns:
            sort_cols = [
                "evidence_arbiter_state_priority",
                "evidence_arbiter_priority",
                "experimental_readiness_priority" if "experimental_readiness_priority" in fallback.columns else "structure_augmented_score",
                "predicted_pIC50",
                "max_market_similarity",
            ]
            ascending = [True, False, False, False, True]
        elif "experimental_readiness_priority" in fallback.columns:
            sort_cols = ["experimental_readiness_priority", "structure_augmented_score", "predicted_pIC50", "max_market_similarity"]
            ascending = [False, False, False, True]
            if "external_evidence_priority" in fallback.columns:
                sort_cols = ["experimental_readiness_priority", "external_evidence_priority", "structure_augmented_score", "predicted_pIC50", "max_market_similarity"]
                ascending = [False, False, False, False, True]
            if "cross_database_consensus_score" in fallback.columns:
                sort_cols = [
                    "experimental_readiness_priority",
                    "external_evidence_priority" if "external_evidence_priority" in fallback.columns else "cross_database_consensus_score",
                    "cross_database_consensus_score",
                    "structure_augmented_score",
                    "predicted_pIC50",
                    "max_market_similarity",
                ]
                ascending = [False, False, False, False, False, True]
        elif "feasible_priority_score" in fallback.columns:
            sort_cols = ["feasible_priority_score", "final_score", "predicted_pIC50", "max_market_similarity"]
            ascending = [False, False, False, True]
        elif "docking_rescore" in fallback.columns:
            sort_cols = ["docking_rescore", "final_score", "predicted_pIC50", "max_market_similarity"]
            ascending = [False, False, False, True]
        shortlist = fallback.sort_values(sort_cols, ascending=ascending).head(25).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "market_comparable_novel_shortlist.csv"
    shortlist.to_csv(out_path, index=False)

    print(f"[OK] Saved shortlist: {out_path}")
    print(f"[INFO] Market median final_score = {market_score_ref:.3f}")
    print(f"[INFO] Market median predicted_pIC50 = {market_pic50_ref:.3f}")
    if market_docking_ref is not None:
        print(f"[INFO] Market median docking_rescore = {market_docking_ref:.3f}")
    if market_interaction_ref is not None:
        print(f"[INFO] Market median interaction support = {market_interaction_ref:.3f}")
    print(shortlist.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
