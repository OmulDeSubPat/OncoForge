from __future__ import annotations

import pandas as pd

from src.agents.multi_agent import add_multiobjective_ranking, build_default_scorer
from src.config import PROJECT_ROOT
from src.models.predict_and_score import score_molecule
from src.structure.docking_rescoring import StructuralConsensusRescorer
from src.structure.interaction_analysis import PoseInteractionAnalyzer


def main() -> None:
    benchmark_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")

    df = pd.read_csv(benchmark_path, low_memory=False)
    scorer = build_default_scorer()
    rescorer = StructuralConsensusRescorer(
        backend="auto",
        pose_dir=PROJECT_ROOT / "reports" / "vina_poses" / "marketed_drugs",
        vina_cpu=1,
        vina_exhaustiveness=6,
        vina_num_modes=5,
    )
    analyzer = PoseInteractionAnalyzer()

    rows = []
    for idx, row in df.iterrows():
        smiles = str(row.get("smiles", "")).strip()
        if not smiles:
            continue
        scored = score_molecule(smiles, scorer=scorer)
        scored["name"] = row.get("name", f"marketed_{idx+1}")
        scored["class"] = row.get("class", "marketed")
        structural = rescorer.score_smiles(smiles, ligand_name=f"marketed_{idx+1:03d}_{scored['name']}")
        scored.update(structural)
        pose_path = structural.get("docking_pose_path")
        if isinstance(pose_path, str) and pose_path:
            scored.update(analyzer.analyze_pose(pose_path, smiles=smiles))
        else:
            scored["interaction_support_score"] = 0.0
            scored["interaction_key_residue_count"] = 0
            scored["interaction_key_residues"] = None
            scored["interaction_top_residues"] = None
            scored["interaction_summary"] = None
        rows.append(scored)

    out = add_multiobjective_ranking(pd.DataFrame(rows), policy=scorer.policy)
    interaction_series = out["interaction_support_score"].astype(float) if "interaction_support_score" in out.columns else 0.0
    out["market_structural_benchmark_score"] = (
        out["final_score"].astype(float)
        + 0.75 * out["docking_rescore"].astype(float)
        + 0.60 * interaction_series
    )
    out = out.sort_values(
        ["market_structural_benchmark_score", "docking_rescore", "predicted_pIC50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["market_structural_rank"] = out.index + 1

    out_path = PROJECT_ROOT / "reports" / "marketed_egfr_structural_benchmark.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved marketed structural benchmark: {out_path}")
    print(
        out[
            [
                "name",
                "predicted_pIC50",
                "final_score",
                "vina_affinity_kcal",
                "docking_rescore",
                "interaction_support_score",
                "market_structural_benchmark_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
