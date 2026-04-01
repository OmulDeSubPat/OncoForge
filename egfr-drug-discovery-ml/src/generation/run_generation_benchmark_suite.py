from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.generation.generation_benchmark import summarize_generated_frame
from src.generation.lineage_tracking import add_parent_child_tracking


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _preferred_generation_artifact(base_path: Path) -> Path:
    candidates = [
        base_path.with_name(base_path.stem + "_structural_crossdb.csv"),
        base_path.with_name(base_path.stem + "_crossdb.csv"),
        base_path.with_name(base_path.stem + "_structural_feasibility.csv"),
        base_path.with_name(base_path.stem + "_feasibility.csv"),
        base_path.with_name(base_path.stem + "_structural_rescored.csv"),
        base_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_path


def _backfill_missing_generation_metadata(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    key_column: str = "smiles",
) -> pd.DataFrame:
    if target_df.empty or source_df.empty:
        return target_df.copy()
    if key_column not in target_df.columns or key_column not in source_df.columns:
        return target_df.copy()

    target = target_df.drop_duplicates(subset=[key_column], keep="first").copy()
    source = source_df.drop_duplicates(subset=[key_column], keep="first").copy()
    target_indexed = target.set_index(key_column)
    source_indexed = source.set_index(key_column).reindex(target_indexed.index)
    combined = target_indexed.combine_first(source_indexed)
    combined.index.name = key_column
    return combined.reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the generator benchmark suite across major generated artifacts.")
    parser.add_argument("--rewrite-inputs", action="store_true", help="Rewrite input CSVs after parent-to-child augmentation.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "generation_benchmark_suite.csv"),
    )
    args = parser.parse_args(argv)

    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    parent_reference = pd.read_csv(ranked_path, low_memory=False) if ranked_path.exists() else pd.DataFrame(columns=["smiles"])
    benchmark_specs = [
        ("generated_analogs_ranked", PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"),
        ("ai_guided_analogs", PROJECT_ROOT / "reports" / "ai_guided_analogs.csv"),
        ("iterative_ai_optimized_candidates", PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"),
    ]

    rows: list[dict] = []
    for benchmark_name, path in benchmark_specs:
        chosen_path = _preferred_generation_artifact(path)
        if not chosen_path.exists():
            continue
        df = pd.read_csv(chosen_path, low_memory=False)
        base_df = df if chosen_path == path else pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
        augmented = add_parent_child_tracking(df, parent_reference=parent_reference)
        if not base_df.empty and chosen_path != path:
            augmented = _backfill_missing_generation_metadata(augmented, base_df)
        if args.rewrite_inputs:
            augmented.to_csv(chosen_path, index=False)
        summary = summarize_generated_frame(
            augmented,
            benchmark_name=benchmark_name,
            out_path=chosen_path.with_suffix(".summary.json"),
            extra={"artifact_path": str(chosen_path)},
        )
        rows.append(summary)

    out = pd.DataFrame(rows)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved generator benchmark suite: {out_path}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
