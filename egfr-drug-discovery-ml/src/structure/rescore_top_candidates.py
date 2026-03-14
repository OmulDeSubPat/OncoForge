from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.structure.docking_rescoring import ReferenceLigandRescorer


def _resolve_output_path(output_arg: str | None, input_path: Path) -> Path:
    if output_arg:
        out_path = Path(output_arg)
        return out_path if out_path.is_absolute() else PROJECT_ROOT / out_path
    return input_path.with_name(f"{input_path.stem}_structural_rescored.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rescore top-ranked candidates with 3D reference-ligand docking proxies.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"),
        help="Input CSV containing at least smiles and final_score columns.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <input>_structural_rescored.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Only rescore the top-k rows by sort column.",
    )
    parser.add_argument(
        "--sort-column",
        type=str,
        default="final_score",
        help="Column used to select top candidates before rescoring.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input candidate file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    required_columns = {"smiles", args.sort_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    candidate_df = df.sort_values(args.sort_column, ascending=False).head(max(1, int(args.top_k))).copy()
    rescorer = ReferenceLigandRescorer()
    if not rescorer.is_available():
        raise RuntimeError("Reference ligand rescoring is unavailable because no marketed reference poses could be prepared.")

    rescored_rows = []
    for _, row in candidate_df.iterrows():
        out_row = row.to_dict()
        out_row.update(rescorer.score_smiles(str(row["smiles"])))
        out_row["structural_priority_score"] = float(out_row.get("final_score", 0.0)) + 0.75 * float(out_row["docking_rescore"])
        rescored_rows.append(out_row)

    out = pd.DataFrame(rescored_rows)
    out = out.sort_values(
        ["structural_priority_score", "docking_rescore", "final_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["structural_rank"] = out.index + 1

    out_path = _resolve_output_path(args.out, input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved structurally rescored candidates: {out_path}")
    print(
        out[
            [
                "smiles",
                "closest_pose_reference",
                "docking_rescore",
                "shape_similarity",
                "usr_similarity",
                "final_score",
                "structural_priority_score",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
