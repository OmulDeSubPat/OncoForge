from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


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
    top_df = working.head(min(top_k, len(working))).copy()

    def _mean(column: str) -> float:
        if column not in working.columns:
            return 0.0
        return float(pd.to_numeric(working[column], errors="coerce").fillna(0.0).mean())

    def _top_mean(column: str) -> float:
        if column not in top_df.columns:
            return 0.0
        return float(pd.to_numeric(top_df[column], errors="coerce").fillna(0.0).mean())

    summary.update(
        {
            "unique_actions": int(working.get("action_name", pd.Series(dtype=str)).nunique()),
            "unique_categories": int(working.get("action_category", pd.Series(dtype=str)).nunique()),
            "mean_predicted_pIC50": _mean("predicted_pIC50"),
            "mean_QED": _mean("QED"),
            "mean_final_score": _mean("final_score"),
            "mean_reward_hacking_risk": _mean("reward_hacking_risk"),
            "mean_generator_priority_score": _mean("generator_priority_score"),
            "mean_parent_similarity": _mean("parent_similarity"),
            "mean_medchem_realism_score": _mean("medchem_realism_score"),
            "mean_synthetic_feasibility_score": _mean("synthetic_feasibility_score"),
            "mean_property_support_score": _mean("property_support_score"),
            "audit_pass_rate": float((working.get("audit_status", pd.Series(dtype=str)) == "pass").mean()),
            "veto_rate": float(pd.to_numeric(working.get("veto", pd.Series(dtype=bool)), errors="coerce").fillna(False).astype(bool).mean()),
            "top_mean_final_score": _top_mean("final_score"),
            "top_mean_predicted_pIC50": _top_mean("predicted_pIC50"),
            "top_mean_generator_priority_score": _top_mean("generator_priority_score"),
            "top_audit_pass_rate": float((top_df.get("audit_status", pd.Series(dtype=str)) == "pass").mean()),
        }
    )

    if "action_category" in working.columns:
        summary["category_mix"] = dict(Counter(working["action_category"].dropna().astype(str)))
    if "reaction_family" in working.columns:
        summary["reaction_family_mix"] = dict(Counter(working["reaction_family"].dropna().astype(str).head(500)))

    if extra:
        summary.update(extra)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
