from __future__ import annotations

import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.multi_agent import add_structure_agent_ranking
from src.config import PROJECT_ROOT
from src.feasibility.experimental_readiness import add_experimental_readiness
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.similarity import morgan_fp, tanimoto_similarity


def main():
    preferred_paths = [
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv",
    ]
    candidates_path = next((path for path in preferred_paths if path.exists()), preferred_paths[-1])
    preferred_market_paths = [
        PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv",
        PROJECT_ROOT / "reports" / "marketed_egfr_scored.csv",
    ]
    market_path = next((path for path in preferred_market_paths if path.exists()), preferred_market_paths[-1])

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidates file: {candidates_path}\n"
            "Run: python -m src.generation.iterative_ai_optimizer"
        )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Missing market benchmark file: {market_path}\n"
            "Run: python -m src.benchmark.score_marketed_egfr"
        )

    cand = load_csv_artifact(
        candidates_path,
        required_columns=["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "audit_status", "final_score"],
        producer="python -m src.generation.iterative_ai_optimizer",
    ).copy()
    market = load_csv_artifact(
        market_path,
        required_columns=["name", "smiles", "predicted_pIC50", "final_score"],
        producer="python -m src.structure.dock_marketed_egfr",
    ).copy()

    market_fps = []
    for _, row in market.iterrows():
        fp = morgan_fp(smiles=row["smiles"])
        if fp is not None:
            market_fps.append((row["name"], row["smiles"], fp))

    rows = []
    for _, row in cand.iterrows():
        cfp = morgan_fp(smiles=row["smiles"])
        if cfp is None:
            continue

        best_name = None
        best_smiles = None
        best_sim = -1.0

        for market_name, market_smiles, market_fp in market_fps:
            sim = tanimoto_similarity(cfp, market_fp)
            if sim > best_sim:
                best_sim = sim
                best_name = market_name
                best_smiles = market_smiles

        out_row = row.to_dict()
        out_row["closest_market_drug"] = best_name
        out_row["closest_market_smiles"] = best_smiles
        out_row["max_market_similarity"] = best_sim
        out_row["market_novelty_score"] = max(0.0, 1.0 - best_sim)
        if "docking_rescore" in market.columns and best_name is not None:
            market_match = market.loc[market["name"] == best_name].head(1)
            if not market_match.empty:
                if "docking_rescore" in market_match.columns:
                    out_row["closest_market_docking_rescore"] = float(market_match.iloc[0]["docking_rescore"])
                if "interaction_support_score" in market_match.columns:
                    out_row["closest_market_interaction_support"] = float(market_match.iloc[0]["interaction_support_score"])
        rows.append(out_row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_experimental_readiness(out, market_df=market)
        out = add_structure_agent_ranking(out)
        out = add_evidence_arbiter_ranking(out)

    if "evidence_arbiter_priority" in out.columns:
        sort_cols = [
            "evidence_arbiter_state_priority",
            "evidence_arbiter_priority",
            "experimental_readiness_priority" if "experimental_readiness_priority" in out.columns else "structure_augmented_score",
            "structure_augmented_score" if "structure_augmented_score" in out.columns else "final_score",
            "final_score",
            "max_market_similarity",
        ]
        ascending = [True, False, False, False, False, True]
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "experimental_readiness_priority" in out.columns:
        sort_cols = ["experimental_readiness_priority", "structure_augmented_score", "final_score", "max_market_similarity"]
        ascending = [False, False, False, True]
        if "external_evidence_priority" in out.columns:
            sort_cols = ["experimental_readiness_priority", "external_evidence_priority", "structure_augmented_score", "final_score", "max_market_similarity"]
            ascending = [False, False, False, False, True]
        if "cross_database_consensus_score" in out.columns:
            sort_cols = [
                "experimental_readiness_priority",
                "external_evidence_priority" if "external_evidence_priority" in out.columns else "cross_database_consensus_score",
                "cross_database_consensus_score",
                "structure_augmented_score",
                "final_score",
                "max_market_similarity",
            ]
            ascending = [False, False, False, False, False, True]
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "feasible_priority_score" in out.columns:
        sort_cols = ["feasible_priority_score", "final_score", "max_market_similarity"]
        ascending = [False, False, True]
        if "docking_rescore" in out.columns:
            sort_cols = ["feasible_priority_score", "docking_rescore", "final_score", "max_market_similarity"]
            ascending = [False, False, False, True]
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "docking_rescore" in out.columns:
        out["structural_priority_score"] = out["final_score"] + 0.75 * out["docking_rescore"]
        out = out.sort_values(
            ["structural_priority_score", "docking_rescore", "final_score", "max_market_similarity"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        out = out.sort_values(
            ["final_score", "max_market_similarity"],
            ascending=[False, True],
        ).reset_index(drop=True)

    out_path = PROJECT_ROOT / "reports" / "candidates_vs_market.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved candidate vs market comparison: {out_path}")
    print(
        out[
            [
                "smiles",
                "predicted_pIC50",
                "QED",
                "reward_hacking_risk",
                "final_score",
                "closest_market_drug",
                "max_market_similarity",
                "round",
            ]
        ].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main()
