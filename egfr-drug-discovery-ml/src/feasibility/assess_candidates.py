from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT
from src.feasibility.assessor import FeasibilityAssessor
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _plot_feasibility(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["feasibility_score"],
        df["predicted_pIC50"],
        c=df["QED"] if "QED" in df.columns else df["feasibility_score"],
        cmap="viridis",
        alpha=0.75,
        s=24,
    )
    ax.axvline(0.60, linestyle="--", color="#6c757d")
    ax.set_xlabel("Feasibility score")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("Candidate Feasibility vs Predicted Potency")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("QED")
    fig.tight_layout()
    fig.savefig(out_dir / "feasibility_vs_potency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assess feasibility evidence for generated candidates.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_feasibility.csv"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=150,
        help="Assess only the top-k rows by final_score when the input is large.",
    )
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "final_score" in df.columns:
        df = df.sort_values("final_score", ascending=False).head(max(1, int(args.top_k))).copy()
    assessor = FeasibilityAssessor()

    rows = []
    for _, row in df.iterrows():
        feasibility = assessor.assess(
            str(row["smiles"]),
            parent_smiles=str(row["parent_seed"]) if "parent_seed" in row and pd.notna(row["parent_seed"]) else None,
            action_name=str(row["action_name"]) if "action_name" in row and pd.notna(row["action_name"]) else None,
            synthetic_feasibility_score=float(row["synthetic_feasibility_score"]) if "synthetic_feasibility_score" in row and pd.notna(row["synthetic_feasibility_score"]) else None,
            medchem_realism_score=float(row["medchem_realism_score"]) if "medchem_realism_score" in row and pd.notna(row["medchem_realism_score"]) else None,
            transformation_confidence=float(row["transformation_confidence_score"]) if "transformation_confidence_score" in row and pd.notna(row["transformation_confidence_score"]) else None,
            reaction_family=str(row["reaction_family"]) if "reaction_family" in row and pd.notna(row["reaction_family"]) else None,
            docking_rescore=float(row["docking_rescore"]) if "docking_rescore" in row and pd.notna(row["docking_rescore"]) else None,
            interaction_support_score=float(row["interaction_support_score"]) if "interaction_support_score" in row and pd.notna(row["interaction_support_score"]) else None,
            interaction_key_residue_count=int(row["interaction_key_residue_count"]) if "interaction_key_residue_count" in row and pd.notna(row["interaction_key_residue_count"]) else None,
        )
        out_row = row.to_dict()
        out_row.update(feasibility)
        base_priority = float(
            out_row.get(
                "interaction_priority_score",
                out_row.get("structural_priority_score", out_row.get("final_score", 0.0)),
            )
        )
        out_row["feasible_priority_score"] = base_priority + 1.25 * float(out_row["feasibility_score"])
        rows.append(out_row)

    out = pd.DataFrame(rows)
    out = add_experimental_readiness(out, market_df=load_market_benchmark())
    out["feasibility_priority"] = out["feasibility_status"].map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    out = out.sort_values(
        [
            "feasibility_priority",
            "experimental_readiness_priority" if "experimental_readiness_priority" in out.columns else "feasible_priority_score",
            "feasible_priority_score",
            "predicted_pIC50",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    out["feasible_rank"] = out.index + 1

    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "n_assessed": int(len(out)),
        "pass_rate": float((out["feasibility_status"] == "pass").mean()) if not out.empty else 0.0,
        "review_rate": float((out["feasibility_status"] == "review").mean()) if not out.empty else 0.0,
        "fail_rate": float((out["feasibility_status"] == "fail").mean()) if not out.empty else 0.0,
        "hard_gate_pass_rate": float(out["feasibility_hard_gate_pass"].mean()) if "feasibility_hard_gate_pass" in out.columns and not out.empty else 0.0,
        "mean_feasibility_score": float(out["feasibility_score"].mean()) if not out.empty else 0.0,
        "ready_rate": float((out["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in out.columns and not out.empty else 0.0,
        "supporting_rate": float((out["experimental_readiness_status"] == "supporting").mean()) if "experimental_readiness_status" in out.columns and not out.empty else 0.0,
        "mean_readiness_score": float(out["experimental_readiness_score"].mean()) if "experimental_readiness_score" in out.columns and not out.empty else 0.0,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_feasibility(out, out_path.parent)
    specific_plot = out_path.with_name(f"{out_path.stem}_feasibility_vs_potency.png")
    generic_plot = out_path.parent / "feasibility_vs_potency.png"
    if generic_plot.exists():
        specific_plot.write_bytes(generic_plot.read_bytes())

    print(f"[OK] Saved feasibility assessment: {out_path}")
    print(f"[OK] Saved feasibility summary: {summary_path}")
    print(
        out[
            [
                "smiles",
                "predicted_pIC50",
                "QED",
                "feasibility_score",
                "feasibility_status",
                "experimental_readiness_score",
                "experimental_readiness_status",
                "max_active_similarity",
                "fragment_support_ratio",
                "feasible_priority_score",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
