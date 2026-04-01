from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.data.dataset_registry import resolve_preferred_processed_dataset


TRACKED_METRICS = [
    "predicted_pIC50",
    "QED",
    "final_score",
    "verified_reward",
    "generator_priority_score",
    "adaptive_action_prior",
    "reward_hacking_risk",
    "cross_database_consensus_score",
    "external_evidence_support",
    "docking_rescore",
    "interaction_support_score",
    "structural_guidance_score",
    "feasibility_score",
    "experimental_readiness_score",
    "evidence_arbiter_support",
]


@lru_cache(maxsize=1)
def _load_ranked_parent_frame() -> pd.DataFrame:
    dataset_path = resolve_preferred_processed_dataset()
    if not dataset_path.exists():
        return pd.DataFrame(columns=["smiles"])
    df = pd.read_csv(dataset_path, low_memory=False)
    if "smiles_canonical" not in df.columns:
        return pd.DataFrame(columns=["smiles"])
    out = pd.DataFrame({"smiles": df["smiles_canonical"].dropna().astype(str).drop_duplicates()})
    if "pIC50_median" in df.columns:
        out["predicted_pIC50"] = pd.to_numeric(df["pIC50_median"], errors="coerce")
    return out.drop_duplicates(subset=["smiles"]).reset_index(drop=True)


def _prepare_lookup(df: pd.DataFrame, key_column: str, prefix: str) -> pd.DataFrame:
    if key_column not in df.columns:
        return pd.DataFrame(columns=[key_column])
    keep_cols = [key_column]
    for column in TRACKED_METRICS:
        if column in df.columns:
            keep_cols.append(column)
    out = df[keep_cols].copy().drop_duplicates(subset=[key_column])
    rename_map = {key_column: prefix.rstrip("_")}
    rename_map.update({column: f"{prefix}{column}" for column in keep_cols if column != key_column})
    return out.rename(columns=rename_map)


def add_parent_child_tracking(
    df: pd.DataFrame,
    *,
    parent_reference: pd.DataFrame | None = None,
    parent_column: str = "parent_seed",
    ancestor_column: str = "ancestor_seed",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "lineage_depth" not in out.columns:
        out["lineage_depth"] = 1 if parent_column in out.columns else 0
    out["lineage_depth"] = pd.to_numeric(out["lineage_depth"], errors="coerce").fillna(0).astype(int)

    if parent_reference is None:
        parent_reference = pd.DataFrame(columns=["smiles"])

    ranked_like = parent_reference.copy()
    if ranked_like.empty:
        ranked_like = _load_ranked_parent_frame()

    if parent_column in out.columns:
        parent_lookup = _prepare_lookup(parent_reference if not parent_reference.empty else ranked_like, "smiles", "parent_")
        if not parent_lookup.empty:
            out = out.merge(parent_lookup, left_on=parent_column, right_on="parent", how="left")
            out = out.drop(columns=["parent"], errors="ignore")

    if ancestor_column in out.columns:
        ancestor_lookup = _prepare_lookup(parent_reference if not parent_reference.empty else ranked_like, "smiles", "ancestor_")
        if not ancestor_lookup.empty:
            out = out.merge(ancestor_lookup, left_on=ancestor_column, right_on="ancestor", how="left")
            out = out.drop(columns=["ancestor"], errors="ignore")

    delta_specs = [
        ("predicted_pIC50", "parent_predicted_pIC50", "delta_predicted_pIC50"),
        ("QED", "parent_QED", "delta_QED"),
        ("final_score", "parent_final_score", "delta_final_score"),
        ("verified_reward", "parent_verified_reward", "delta_verified_reward"),
        ("generator_priority_score", "parent_generator_priority_score", "delta_generator_priority_score"),
        ("feasibility_score", "parent_feasibility_score", "delta_feasibility_score"),
        ("docking_rescore", "parent_docking_rescore", "delta_docking_rescore"),
    ]
    for child_col, parent_col_name, delta_col in delta_specs:
        if child_col in out.columns and parent_col_name in out.columns:
            out[delta_col] = (
                pd.to_numeric(out[child_col], errors="coerce")
                - pd.to_numeric(out[parent_col_name], errors="coerce")
            )

    if "delta_final_score" in out.columns:
        out["improved_over_parent_final_score"] = (pd.to_numeric(out["delta_final_score"], errors="coerce") > 0).fillna(False)
    if "delta_predicted_pIC50" in out.columns:
        out["improved_over_parent_potency"] = (pd.to_numeric(out["delta_predicted_pIC50"], errors="coerce") > 0).fillna(False)
    if "delta_QED" in out.columns:
        out["improved_over_parent_qed"] = (pd.to_numeric(out["delta_QED"], errors="coerce") > 0).fillna(False)
    if "delta_verified_reward" in out.columns:
        out["improved_over_parent_verified_reward"] = (pd.to_numeric(out["delta_verified_reward"], errors="coerce") > 0).fillna(False)

    improvement_flags = [
        column
        for column in [
            "improved_over_parent_final_score",
            "improved_over_parent_potency",
            "improved_over_parent_qed",
            "improved_over_parent_verified_reward",
        ]
        if column in out.columns
    ]
    if improvement_flags:
        out["parent_improvement_count"] = out[improvement_flags].astype(int).sum(axis=1)

    if ancestor_column in out.columns:
        out["lineage_root"] = out[ancestor_column].fillna(out.get(parent_column, out.get("smiles")))
    elif parent_column in out.columns:
        out["lineage_root"] = out[parent_column].fillna(out.get("smiles"))
    else:
        out["lineage_root"] = out.get("smiles")

    if "lineage_path" not in out.columns:
        if parent_column in out.columns:
            out["lineage_path"] = out[parent_column].fillna("") + " -> " + out["smiles"].fillna("")
        else:
            out["lineage_path"] = out.get("smiles", pd.Series(dtype=str)).fillna("")

    return out
