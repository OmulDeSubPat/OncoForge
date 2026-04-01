from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.generation.generation_benchmark import summarize_generated_frame
from src.generation.lineage_tracking import add_parent_child_tracking
from src.generation.medchem_mutations import generate_medchem_outcomes
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.chem import canonicalize_smiles


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate and score medicinal-chemistry analogs from top-ranked seeds.")
    parser.add_argument("--seed-count", type=int, default=24, help="Number of ranked seed molecules to expand.")
    parser.add_argument("--variants-per-seed", type=int, default=60, help="Maximum number of med-chem variants per seed.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "generated_analogs_ranked.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(
            f"Missing ranked dataset: {ranked_path}\n"
            "Run: python -m src.models.rank_dataset"
        )

    ranked_df = load_csv_artifact(
        ranked_path,
        required_columns=["smiles", "veto", "audit_pass", "reward_hacking_risk", "agent_disagreement_score", "applicability_score", "final_score"],
        producer="python -m src.models.rank_dataset",
    )
    scorer = build_default_scorer()

    seed_df = ranked_df[
        (ranked_df["veto"] == False)
        & (ranked_df["audit_pass"] == True)
        & (ranked_df["reward_hacking_risk"] <= 0.35)
        & (ranked_df["agent_disagreement_score"] <= 0.45)
        & (ranked_df["applicability_score"] >= 0.30)
    ].head(int(args.seed_count))
    seed_smiles = seed_df["smiles"].tolist()

    generated_pairs = []
    seen = set()
    attempted_candidates = 0

    print(f"[INFO] Using {len(seed_smiles)} high-confidence seed molecules")
    for seed in tqdm(seed_smiles, desc="Generating analogs"):
        analogs = generate_medchem_outcomes(seed, max_variants=int(args.variants_per_seed))
        attempted_candidates += len(analogs)

        for analog in analogs:
            canonical_smiles = canonicalize_smiles(analog.smiles)
            if not canonical_smiles or canonical_smiles in seen:
                continue
            seen.add(canonical_smiles)
            generated_pairs.append(
                {
                    "smiles": canonical_smiles,
                    "parent_seed": seed,
                    "action_name": analog.action_name,
                    "action_category": analog.category,
                    "action_rule_source": analog.rule_source,
                    "reaction_family": analog.reaction_family,
                    "synthetic_route": analog.synthetic_route,
                    "synthetic_feasibility_score": analog.synthetic_feasibility_score,
                    "medchem_realism_score": analog.medchem_realism_score,
                    "transformation_confidence_score": analog.transformation_confidence,
                    "preserves_scaffold": analog.preserves_scaffold,
                    "parent_similarity": analog.parent_similarity,
                    "property_support_score": analog.property_support_score,
                    "category_priority_score": analog.category_priority_score,
                    "generator_priority_score": analog.generator_priority_score,
                    "hard_constraint_pass": analog.hard_constraint_pass,
                    "hard_constraint_notes": analog.hard_constraint_notes,
                    "introduced_warhead": analog.introduced_warhead,
                    "warhead_retained": analog.warhead_retained,
                    "alert_count": analog.alert_count,
                    "severe_alert_count": analog.severe_alert_count,
                    "structural_guidance_score": analog.structural_guidance_score,
                    "structure_guidance_reference": analog.structure_guidance_reference,
                    "structure_guidance_backend": analog.structure_guidance_backend,
                    "adaptive_action_prior": analog.adaptive_action_prior,
                    "ancestor_seed": seed,
                    "lineage_depth": 1,
                    "lineage_path": f"{seed} -> {canonical_smiles}",
                }
            )

    if not generated_pairs:
        print("[WARN] No valid analogs generated.")
        return

    generated_df = pd.DataFrame(generated_pairs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    out = score_smiles_list(generated_df["smiles"].tolist(), scorer=scorer)
    out = out.merge(generated_df, on="smiles", how="left")
    out = add_parent_child_tracking(out, parent_reference=ranked_df)
    out["generator_composite_score"] = (
        _series(out, "final_score")
        + 0.70 * _series(out, "generator_priority_score")
        + 0.20 * _series(out, "parent_similarity")
        + 0.10 * _series(out, "property_support_score")
        + 0.20 * _series(out, "structural_guidance_score")
        + 0.18 * _series(out, "adaptive_action_prior")
        - 0.25 * _series(out, "reward_hacking_risk")
    )
    out = out.sort_values(
        ["generator_composite_score", "final_score", "predicted_pIC50", "QED"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = pd.io.common.stringify_path(args.out)
    out.to_csv(out_path, index=False)
    summarize_generated_frame(
        out,
        benchmark_name="generated_analogs_ranked",
        out_path=Path(out_path).with_suffix(".summary.json"),
        extra={
            "seed_count": int(len(seed_smiles)),
            "variants_per_seed": int(args.variants_per_seed),
            "attempted_candidates": int(attempted_candidates),
        },
    )

    print(f"[OK] Saved generated analogs: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
