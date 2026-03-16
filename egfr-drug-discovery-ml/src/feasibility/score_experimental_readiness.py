from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score experimental-readiness evidence for candidate molecules.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv"),
    )
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    validator = CrossDatabaseValidator()
    validated = validator.validate_frame(df)
    validated = add_external_evidence_agent_ranking(validated)
    out = add_experimental_readiness(validated, market_df=load_market_benchmark())
    out = add_evidence_arbiter_ranking(out)

    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "n_candidates": int(len(out)),
        "ready_rate": float((out["experimental_readiness_status"] == "ready").mean()) if not out.empty else 0.0,
        "supporting_rate": float((out["experimental_readiness_status"] == "supporting").mean()) if not out.empty else 0.0,
        "hold_rate": float((out["experimental_readiness_status"] == "hold").mean()) if not out.empty else 0.0,
        "mean_readiness_score": float(out["experimental_readiness_score"].mean()) if not out.empty else 0.0,
        "mean_evidence_arbiter_support": float(out["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in out.columns and not out.empty else 0.0,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved experimental readiness file: {out_path}")
    print(f"[OK] Saved readiness summary: {summary_path}")
    preview_cols = [
        "smiles",
        "predicted_pIC50",
        "feasibility_score",
        "docking_rescore",
        "interaction_support_score",
        "experimental_readiness_score",
        "experimental_readiness_status",
        "experimental_track",
    ]
    preview_cols = [column for column in preview_cols if column in out.columns]
    print(out[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
