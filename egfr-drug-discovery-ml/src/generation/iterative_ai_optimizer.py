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


def select_seed_pool(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    filtered = df[
        (df["predicted_pIC50"] >= 8.5)
        & (df["QED"] >= 0.40)
        & (df["reward_hacking_risk"] <= 0.30)
        & (df["agent_disagreement_score"] <= 0.45)
        & (df["applicability_score"] >= 0.30)
        & (df["audit_pass"] == True)
        & (df["veto"] == False)
    ].copy()
    return filtered.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Iteratively optimize top-ranked molecules with medicinal-chemistry actions.")
    parser.add_argument("--seed-count", type=int, default=10, help="Initial seed pool size.")
    parser.add_argument("--rounds", type=int, default=4, help="Number of optimization rounds.")
    parser.add_argument("--beam-width", type=int, default=12, help="How many top candidates survive each round.")
    parser.add_argument("--variants-per-seed", type=int, default=80, help="Maximum variants per seed in each round.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"),
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
        required_columns=["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "agent_disagreement_score", "applicability_score", "audit_pass", "veto", "final_score"],
        producer="python -m src.models.rank_dataset",
    )
    scorer = build_default_scorer()

    current_pool = select_seed_pool(ranked_df, top_k=int(args.seed_count))
    current_pool["ancestor_seed"] = current_pool["smiles"]
    current_pool["lineage_depth"] = 0
    current_pool["lineage_path"] = current_pool["smiles"]
    all_generated = []
    seen = set(current_pool["smiles"].tolist())
    attempted_candidates = 0

    n_rounds = int(args.rounds)
    beam_width = int(args.beam_width)
    variants_per_seed = int(args.variants_per_seed)

    print(f"[INFO] Starting iterative optimization with {len(current_pool)} seed molecules")

    for round_idx in range(1, n_rounds + 1):
        print(f"\n[INFO] Round {round_idx}")

        candidate_pairs = []

        for _, row in tqdm(current_pool.iterrows(), total=len(current_pool), desc=f"Round {round_idx} generation"):
            parent_smiles = row["smiles"]
            variants = generate_medchem_outcomes(parent_smiles, max_variants=variants_per_seed)
            attempted_candidates += len(variants)
            ancestor_seed = str(row.get("ancestor_seed", parent_smiles))
            lineage_depth = int(row.get("lineage_depth", 0) or 0)
            lineage_path = str(row.get("lineage_path", parent_smiles))

            for variant in variants:
                canonical_smiles = canonicalize_smiles(variant.smiles)
                if not canonical_smiles or canonical_smiles in seen:
                    continue
                seen.add(canonical_smiles)
                candidate_pairs.append(
                    {
                        "smiles": canonical_smiles,
                        "parent_seed": parent_smiles,
                        "ancestor_seed": ancestor_seed,
                        "lineage_depth": lineage_depth + 1,
                        "lineage_path": f"{lineage_path} -> {canonical_smiles}",
                        "round": round_idx,
                        "action_name": variant.action_name,
                        "action_category": variant.category,
                        "action_rule_source": variant.rule_source,
                        "reaction_family": variant.reaction_family,
                        "synthetic_route": variant.synthetic_route,
                        "synthetic_feasibility_score": variant.synthetic_feasibility_score,
                        "medchem_realism_score": variant.medchem_realism_score,
                        "transformation_confidence_score": variant.transformation_confidence,
                        "preserves_scaffold": variant.preserves_scaffold,
                        "parent_similarity": variant.parent_similarity,
                        "property_support_score": variant.property_support_score,
                        "category_priority_score": variant.category_priority_score,
                        "generator_priority_score": variant.generator_priority_score,
                        "adaptive_action_prior": variant.adaptive_action_prior,
                        "hard_constraint_pass": variant.hard_constraint_pass,
                        "hard_constraint_notes": variant.hard_constraint_notes,
                        "introduced_warhead": variant.introduced_warhead,
                        "warhead_retained": variant.warhead_retained,
                        "alert_count": variant.alert_count,
                        "severe_alert_count": variant.severe_alert_count,
                        "structural_guidance_score": variant.structural_guidance_score,
                        "structure_guidance_reference": variant.structure_guidance_reference,
                        "structure_guidance_backend": variant.structure_guidance_backend,
                    }
                )

        if not candidate_pairs:
            print("[WARN] No candidates generated in this round.")
            break

        candidate_df = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        cand_df = score_smiles_list(candidate_df["smiles"].tolist(), scorer=scorer)
        cand_df = cand_df.merge(candidate_df, on="smiles", how="left")
        cand_df = add_parent_child_tracking(cand_df, parent_reference=ranked_df)
        cand_df["generator_composite_score"] = (
            _series(cand_df, "final_score")
            + 0.95 * _series(cand_df, "generator_priority_score")
            + 0.22 * _series(cand_df, "adaptive_action_prior")
            + 0.25 * _series(cand_df, "parent_similarity")
            + 0.12 * _series(cand_df, "property_support_score")
            + 0.24 * _series(cand_df, "structural_guidance_score")
            - 0.20 * _series(cand_df, "reward_hacking_risk")
        )
        cand_df = cand_df[
            (cand_df["predicted_pIC50"] >= 8.3)
            & (cand_df["QED"] >= 0.35)
            & (cand_df["reward_hacking_risk"] <= 0.45)
            & (cand_df["agent_disagreement_score"] <= 0.55)
            & (cand_df["audit_status"] != "fail")
            & (cand_df["veto"] == False)
            & (_series(cand_df, "generator_priority_score") >= 0.40)
        ].copy()
        cand_df = cand_df.sort_values(
            ["generator_composite_score", "final_score", "predicted_pIC50", "QED"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        if cand_df.empty:
            print("[WARN] No candidates survived filters.")
            break

        all_generated.append(cand_df)
        current_pool = cand_df.head(beam_width).copy()

        print("[INFO] Top round candidates:")
        print(
            current_pool[
                ["smiles", "predicted_pIC50", "QED", "reward_hacking_risk", "final_score"]
            ].head(10).to_string(index=False)
        )

    if not all_generated:
        print("[WARN] No optimized molecules were generated.")
        return

    out = pd.concat(all_generated, ignore_index=True)
    out = out.drop_duplicates(subset=["smiles"]).sort_values(
        ["generator_composite_score", "final_score", "predicted_pIC50", "QED"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out_path = pd.io.common.stringify_path(args.out)
    out.to_csv(out_path, index=False)
    summarize_generated_frame(
        out,
        benchmark_name="iterative_ai_optimized_candidates",
        out_path=Path(out_path).with_suffix(".summary.json"),
        extra={
            "seed_count": int(args.seed_count),
            "rounds": int(args.rounds),
            "beam_width": int(args.beam_width),
            "variants_per_seed": int(args.variants_per_seed),
            "attempted_candidates": int(attempted_candidates),
        },
    )

    print(f"\n[OK] Saved iterative optimization results: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
