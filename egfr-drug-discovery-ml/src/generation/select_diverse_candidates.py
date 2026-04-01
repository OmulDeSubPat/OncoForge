from __future__ import annotations

import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.multi_agent import add_structure_agent_ranking
from src.agents.structure_evidence_arbiter import add_structure_evidence_arbiter
from src.config import PROJECT_ROOT
from src.feasibility.experimental_readiness import add_experimental_readiness
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.similarity import morgan_fp, tanimoto_similarity


def main():
    preferred_paths = [
        PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_crossdb.csv",
        PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_feasibility.csv",
        PROJECT_ROOT / "reports" / "generated_analogs_ranked_structural_rescored.csv",
        PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv",
    ]
    in_path = next((path for path in preferred_paths if path.exists()), preferred_paths[-1])
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
    if "feasibility_status" in df.columns:
        df = df[(df["feasibility_status"] == "pass") & (df["feasibility_score"] >= 0.60)].copy()
    if "experimental_readiness_status" in df.columns:
        df = df[df["experimental_readiness_status"].isin(["ready", "supporting"])].copy()
    if "cross_database_status" in df.columns:
        df = df[df["cross_database_status"] != "weak"].copy()
    if "external_evidence_status" in df.columns:
        df = df[df["external_evidence_status"] != "fail"].copy()
    if "docking_rescore" in df.columns:
        df = df[df["docking_rescore"] >= 0.45].copy()

    if not df.empty:
        df = add_experimental_readiness(df)
        df = add_structure_agent_ranking(df)
        df = add_evidence_arbiter_ranking(df)
        df = add_structure_evidence_arbiter(df)

    if "structure_evidence_priority" in df.columns:
        df = df[df["structure_evidence_status"] != "fail"].copy()
        sort_cols = [
            "structure_evidence_state_priority",
            "structure_evidence_pareto_front_rank",
            "structure_evidence_priority",
            "evidence_arbiter_priority" if "evidence_arbiter_priority" in df.columns else "final_score",
            "final_score",
        ]
        ascending = [True, True, False, False, False]
    elif "evidence_arbiter_priority" in df.columns:
        df = df[df["evidence_arbiter_status"] != "fail"].copy()
        sort_cols = [
            "evidence_arbiter_state_priority",
            "evidence_arbiter_priority",
            "experimental_readiness_priority" if "experimental_readiness_priority" in df.columns else "structure_augmented_score",
            "structure_augmented_score" if "structure_augmented_score" in df.columns else "final_score",
            "final_score",
        ]
        ascending = [True, False, False, False, False]
    elif "experimental_readiness_priority" in df.columns:
        sort_cols = ["experimental_readiness_priority", "structure_augmented_score", "final_score"]
        ascending = [False] * len(sort_cols)
        if "external_evidence_priority" in df.columns:
            sort_cols = ["experimental_readiness_priority", "external_evidence_priority", "structure_augmented_score", "final_score"]
            ascending = [False] * len(sort_cols)
        if "cross_database_consensus_score" in df.columns:
            sort_cols = [
                "experimental_readiness_priority",
                "external_evidence_priority" if "external_evidence_priority" in df.columns else "cross_database_consensus_score",
                "cross_database_consensus_score",
                "structure_augmented_score",
                "final_score",
            ]
            ascending = [False] * len(sort_cols)
    elif "feasible_priority_score" in df.columns:
        sort_cols = ["feasible_priority_score", "final_score"]
        ascending = [False] * len(sort_cols)
        if "docking_rescore" in df.columns:
            sort_cols = ["feasible_priority_score", "docking_rescore", "final_score"]
            ascending = [False] * len(sort_cols)
    elif "docking_rescore" in df.columns:
        df = df[df["docking_rescore"] >= 0.45].copy()
        sort_cols = ["structural_priority_score", "docking_rescore", "final_score"]
        ascending = [False] * len(sort_cols)
    else:
        sort_cols = ["final_score"]
        ascending = [False]
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

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
