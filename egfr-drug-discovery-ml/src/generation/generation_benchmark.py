from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import DataStructs

from src.data.dataset_registry import resolve_preferred_processed_dataset
from src.utils.similarity import mol_from_smiles, morgan_fp, murcko_scaffold_smiles


def summarize_generated_frame(
    df: pd.DataFrame,
    benchmark_name: str,
    out_path: Path,
    top_k: int = 100,
    extra: dict | None = None,
) -> dict:
    summary: dict[str, object] = {
        "benchmark_name": benchmark_name,
        "n_candidates": int(len(df)),
    }
    if df.empty:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    working = df.copy()
    if "smiles" in working.columns:
        working["smiles"] = working["smiles"].astype(str)
    if "parent_seed" in working.columns:
        working["parent_seed"] = working["parent_seed"].astype(str)
    top_df = working.head(min(top_k, len(working))).copy()

    def _mean(*columns: str) -> float:
        for column in columns:
            if column in working.columns:
                return float(pd.to_numeric(working[column], errors="coerce").fillna(0.0).mean())
        return 0.0

    def _top_mean(*columns: str) -> float:
        for column in columns:
            if column in top_df.columns:
                return float(pd.to_numeric(top_df[column], errors="coerce").fillna(0.0).mean())
        return 0.0

    def _rate(mask: pd.Series) -> float:
        if mask.empty:
            return 0.0
        return float(mask.fillna(False).astype(bool).mean())

    generated_smiles = working.get("smiles", pd.Series(dtype=str)).dropna().astype(str).tolist()
    unique_smiles = set(generated_smiles)
    n_unique = len(unique_smiles)
    attempted_candidates = int(extra.get("attempted_candidates", 0)) if extra else 0
    reference = _generation_reference()
    novelty_payload = _novelty_metrics(generated_smiles, reference)
    diversity_payload = _diversity_metrics(generated_smiles)
    lineage_depth_series = pd.to_numeric(working.get("lineage_depth", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    max_lineage_depth = lineage_depth_series.max() if not lineage_depth_series.empty else 0.0

    summary.update(
        {
            "n_unique_smiles": int(n_unique),
            "validity_rate": float(n_unique / max(1, attempted_candidates)) if attempted_candidates else 1.0,
            "uniqueness_rate": float(n_unique / max(1, len(working))),
            "unique_actions": int(working.get("action_name", pd.Series(dtype=str)).nunique()),
            "unique_categories": int(working.get("action_category", pd.Series(dtype=str)).nunique()),
            "mean_predicted_pIC50": _mean("predicted_pIC50"),
            "mean_QED": _mean("QED"),
            "mean_final_score": _mean("final_score"),
            "mean_reward_hacking_risk": _mean("reward_hacking_risk"),
            "mean_generator_priority_score": _mean("generator_priority_score"),
            "mean_adaptive_action_prior": _mean("adaptive_action_prior", "parent_adaptive_action_prior", "ancestor_adaptive_action_prior"),
            "mean_parent_similarity": _mean("parent_similarity"),
            "mean_medchem_realism_score": _mean("medchem_realism_score"),
            "mean_synthetic_feasibility_score": _mean("synthetic_feasibility_score"),
            "mean_property_support_score": _mean("property_support_score"),
            "audit_pass_rate": _rate(working.get("audit_status", pd.Series(dtype=str)) == "pass"),
            "audit_review_rate": _rate(working.get("audit_status", pd.Series(dtype=str)) == "review"),
            "veto_rate": float(pd.to_numeric(working.get("veto", pd.Series(dtype=bool)), errors="coerce").fillna(False).astype(bool).mean()),
            "feasibility_pass_rate": _rate(working.get("feasibility_status", pd.Series(dtype=str)) == "pass"),
            "feasibility_review_rate": _rate(working.get("feasibility_status", pd.Series(dtype=str)) == "review"),
            "readiness_ready_rate": _rate(working.get("experimental_readiness_status", pd.Series(dtype=str)) == "ready"),
            "cross_database_pass_rate": _rate(working.get("cross_database_status", pd.Series(dtype=str)) == "strong"),
            "external_evidence_pass_rate": _rate(working.get("external_evidence_status", pd.Series(dtype=str)) == "pass"),
            "evidence_arbiter_pass_rate": _rate(working.get("evidence_arbiter_status", pd.Series(dtype=str)) == "pass"),
            "top_mean_final_score": _top_mean("final_score"),
            "top_mean_predicted_pIC50": _top_mean("predicted_pIC50"),
            "top_mean_generator_priority_score": _top_mean("generator_priority_score"),
            "top_mean_adaptive_action_prior": _top_mean("adaptive_action_prior", "parent_adaptive_action_prior", "ancestor_adaptive_action_prior"),
            "top_audit_pass_rate": _rate(top_df.get("audit_status", pd.Series(dtype=str)) == "pass"),
            "mean_lineage_depth": float(lineage_depth_series.mean()) if not lineage_depth_series.empty else 0.0,
            "max_lineage_depth": int(max_lineage_depth) if pd.notna(max_lineage_depth) else 0,
        }
    )
    summary.update(novelty_payload)
    summary.update(diversity_payload)

    parent_improvement_metrics = {
        "parent_improvement_rate_final_score": "improved_over_parent_final_score",
        "parent_improvement_rate_potency": "improved_over_parent_potency",
        "parent_improvement_rate_qed": "improved_over_parent_qed",
        "parent_improvement_rate_verified_reward": "improved_over_parent_verified_reward",
    }
    for metric_name, column_name in parent_improvement_metrics.items():
        if column_name in working.columns:
            summary[metric_name] = _rate(working[column_name])
    for delta_name in [
        "delta_final_score",
        "delta_predicted_pIC50",
        "delta_QED",
        "delta_verified_reward",
        "delta_feasibility_score",
        "delta_docking_rescore",
    ]:
        if delta_name in working.columns:
            summary[f"mean_{delta_name}"] = float(pd.to_numeric(working[delta_name], errors="coerce").fillna(0.0).mean())
            summary[f"top_mean_{delta_name}"] = float(pd.to_numeric(top_df[delta_name], errors="coerce").fillna(0.0).mean())

    if "preserves_scaffold" in working.columns:
        summary["scaffold_retention_rate"] = _rate(working["preserves_scaffold"])
    adaptive_prior_column = next(
        (column for column in ["adaptive_action_prior", "parent_adaptive_action_prior", "ancestor_adaptive_action_prior"] if column in working.columns),
        None,
    )
    if adaptive_prior_column is not None:
        summary["strong_transformation_memory_rate"] = _rate(pd.to_numeric(working[adaptive_prior_column], errors="coerce").fillna(0.0) >= 0.60)
    if "parent_seed" in working.columns and "smiles" in working.columns:
        summary["exact_parent_reuse_rate"] = _rate(working["smiles"] == working["parent_seed"])
    if "lineage_root" in working.columns:
        summary["lineage_root_count"] = int(working["lineage_root"].dropna().astype(str).nunique())
    lineage_depth_series = pd.to_numeric(working.get("lineage_depth", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    summary["mean_lineage_depth"] = float(lineage_depth_series.mean()) if not lineage_depth_series.empty else 0.0
    max_lineage_depth = lineage_depth_series.max() if not lineage_depth_series.empty else 0.0
    summary["max_lineage_depth"] = int(max_lineage_depth) if pd.notna(max_lineage_depth) else 0

    if "action_category" in working.columns:
        summary["category_mix"] = dict(Counter(working["action_category"].dropna().astype(str)))
    if "reaction_family" in working.columns:
        summary["reaction_family_mix"] = dict(Counter(working["reaction_family"].dropna().astype(str).head(500)))

    if extra:
        summary.update(extra)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@lru_cache(maxsize=1)
def _generation_reference() -> dict[str, object]:
    dataset_path = resolve_preferred_processed_dataset()
    if not dataset_path.exists():
        return {
            "smiles": set(),
            "fps": [],
            "scaffolds": set(),
        }
    df = pd.read_csv(dataset_path, low_memory=False)
    smiles_series = df.get("smiles_canonical", pd.Series(dtype=str)).dropna().astype(str).drop_duplicates()
    fps = []
    scaffolds = set()
    smiles_set = set()
    for smiles in smiles_series.tolist():
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = morgan_fp(mol=mol)
        if fp is None:
            continue
        fps.append(fp)
        smiles_set.add(smiles)
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold:
            scaffolds.add(scaffold)
    return {"smiles": smiles_set, "fps": fps, "scaffolds": scaffolds}


def _novelty_metrics(generated_smiles: list[str], reference: dict[str, object]) -> dict[str, float]:
    if not generated_smiles:
        return {
            "exact_novelty_rate": 0.0,
            "scaffold_novelty_rate": 0.0,
            "mean_max_train_similarity": 0.0,
            "novelty_rate_tanimoto_lt_055": 0.0,
        }
    reference_smiles = reference.get("smiles", set())
    reference_fps = reference.get("fps", [])
    reference_scaffolds = reference.get("scaffolds", set())

    exact_novel = 0
    scaffold_novel = 0
    max_sims: list[float] = []
    tanimoto_novel = 0
    for smiles in generated_smiles:
        if smiles not in reference_smiles:
            exact_novel += 1
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold and scaffold not in reference_scaffolds:
            scaffold_novel += 1
        fp = morgan_fp(smiles=smiles)
        if fp is None or not reference_fps:
            max_sims.append(0.0)
            tanimoto_novel += 1
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(reference_fps))
        max_sim = float(max(sims)) if sims else 0.0
        max_sims.append(max_sim)
        if max_sim < 0.55:
            tanimoto_novel += 1
    return {
        "exact_novelty_rate": float(exact_novel / len(generated_smiles)),
        "scaffold_novelty_rate": float(scaffold_novel / len(generated_smiles)),
        "mean_max_train_similarity": float(sum(max_sims) / len(max_sims)),
        "novelty_rate_tanimoto_lt_055": float(tanimoto_novel / len(generated_smiles)),
    }


def _diversity_metrics(generated_smiles: list[str], max_sample: int = 160) -> dict[str, float]:
    unique_smiles = list(dict.fromkeys(generated_smiles))[:max_sample]
    fps = [morgan_fp(smiles=smiles) for smiles in unique_smiles]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return {"internal_diversity": 0.0}
    pairwise_diversities: list[float] = []
    for idx, fp in enumerate(fps[:-1]):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[idx + 1 :])
        pairwise_diversities.extend(1.0 - float(sim) for sim in sims)
    return {
        "internal_diversity": float(sum(pairwise_diversities) / len(pairwise_diversities)) if pairwise_diversities else 0.0,
    }
