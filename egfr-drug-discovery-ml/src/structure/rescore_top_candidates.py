from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.structure.docking_rescoring import StructuralConsensusRescorer


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
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "reference", "vina"],
        default="auto",
        help="Structural backend to use. 'auto' combines Vina with the reference-ligand proxy when available.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="CPU count passed to AutoDock Vina for each docking job.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=6,
        help="Vina search exhaustiveness. Higher values improve search depth but increase runtime.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=5,
        help="Maximum number of Vina poses to retain for each ligand.",
    )
    parser.add_argument(
        "--pose-dir",
        type=str,
        default=None,
        help="Optional directory where docked ligand poses are saved when Vina is used.",
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
    pose_dir = None
    if args.pose_dir:
        pose_dir = Path(args.pose_dir)
        if not pose_dir.is_absolute():
            pose_dir = PROJECT_ROOT / pose_dir
    rescorer = StructuralConsensusRescorer(
        backend=args.backend,
        pose_dir=pose_dir,
        vina_cpu=args.cpu,
        vina_exhaustiveness=args.exhaustiveness,
        vina_num_modes=args.num_modes,
    )
    if not rescorer.is_available():
        raise RuntimeError("Structural rescoring is unavailable because no supported backend could be initialized.")

    rescored_rows = []
    for row_index, (_, row) in enumerate(candidate_df.iterrows(), start=1):
        out_row = row.to_dict()
        ligand_tokens = [f"row_{row_index:03d}"]
        if "feasible_rank" in row and pd.notna(row["feasible_rank"]):
            ligand_tokens.append(f"feasible_{int(row['feasible_rank']):03d}")
        elif "rank" in row and pd.notna(row["rank"]):
            ligand_tokens.append(f"rank_{int(row['rank']):03d}")
        ligand_name = "_".join(ligand_tokens)
        out_row.update(rescorer.score_smiles(str(row["smiles"]), ligand_name=ligand_name))
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

    preview_columns = [
        "smiles",
        "docking_backend",
        "docking_rescore",
        "vina_affinity_kcal",
        "reference_docking_rescore",
        "final_score",
        "structural_priority_score",
    ]
    preview_columns = [column for column in preview_columns if column in out.columns]

    print(f"[OK] Saved structurally rescored candidates: {out_path}")
    print(out[preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
