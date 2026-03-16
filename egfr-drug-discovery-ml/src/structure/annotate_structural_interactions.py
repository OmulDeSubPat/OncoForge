from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.structure.interaction_analysis import PoseInteractionAnalyzer


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Annotate docking-backed candidate files with residue interaction evidence.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    out_path = _resolve_path(args.out) if args.out else input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Missing structural file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "docking_pose_path" not in df.columns:
        raise ValueError(f"{input_path} is missing the docking_pose_path column needed for interaction annotation.")

    analyzer = PoseInteractionAnalyzer()
    rows = []
    for _, row in df.iterrows():
        out_row = row.to_dict()
        pose_path = row.get("docking_pose_path")
        if isinstance(pose_path, str) and pose_path.strip() and Path(pose_path).exists():
            out_row.update(analyzer.analyze_pose(pose_path, smiles=str(row.get("smiles", ""))))
        else:
            out_row.update(
                {
                    "interaction_hbond_count": 0,
                    "interaction_hydrophobic_count": 0,
                    "interaction_aromatic_count": 0,
                    "interaction_salt_bridge_count": 0,
                    "interaction_key_residue_count": 0,
                    "interaction_key_residues": None,
                    "interaction_top_residues": None,
                    "interaction_support_score": 0.0,
                    "interaction_hinge_contact": False,
                    "interaction_summary": None,
                }
            )

        base_priority = float(out_row.get("structural_priority_score", out_row.get("final_score", 0.0)))
        out_row["interaction_priority_score"] = base_priority + 0.60 * float(out_row.get("interaction_support_score", 0.0))
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    sort_cols = ["interaction_priority_score", "structural_priority_score", "docking_rescore", "final_score"]
    sort_cols = [column for column in sort_cols if column in out_df.columns]
    out_df = out_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    out_df["interaction_rank"] = out_df.index + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    preview_cols = [
        "smiles",
        "vina_affinity_kcal",
        "interaction_support_score",
        "interaction_key_residues",
        "interaction_top_residues",
        "interaction_priority_score",
    ]
    preview_cols = [column for column in preview_cols if column in out_df.columns]
    print(f"[OK] Saved interaction-annotated structural file: {out_path}")
    print(out_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
