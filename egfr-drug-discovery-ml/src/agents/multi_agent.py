from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from src.config import PROJECT_ROOT
from src.data.dataset_registry import resolve_preferred_processed_dataset
from src.features.descriptor_features import descriptor_vector_from_mol
from src.utils.advanced_filters import (
    covalent_warhead_alerts,
    pains_alert,
    severe_structural_alerts,
    structural_alerts,
)
from src.utils.sa_score import simple_sa_score
from src.utils.similarity import bulk_tanimoto_similarity, mol_from_smiles, morgan_fp, top_k_mean


@dataclass
class ReferenceLibrary:
    smiles: list[str]
    fps: list[Any]
    names: list[str] | None = None


@dataclass(frozen=True)
class ModelBundle:
    name: str
    feature_set: str
    model: Any


@dataclass(frozen=True)
class ScoringPolicy:
    potency_uncertainty_weight: float = 1.50
    out_of_domain_similarity_threshold: float = 0.20
    extreme_novelty_similarity_threshold: float = 0.15
    marketed_copy_penalty_threshold: float = 0.90
    near_copy_similarity_threshold: float = 0.97
    suspicious_potency_threshold: float = 9.0
    suspicious_uncertainty_threshold: float = 0.15
    audit_pass_risk_threshold: float = 0.40
    audit_pass_disagreement_threshold: float = 0.55
    audit_pass_min_applicability: float = 0.25
    qed_veto_threshold: float = 0.20
    sa_veto_threshold: float = 6.5
    uncertainty_veto_threshold: float = 0.35
    multiobjective_weight: float = 1.20
    final_risk_weight: float = 1.50
    final_veto_weight: float = 2.50


def _fingerprint_array(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_join(values: list[str]) -> str | None:
    return ";".join(values) if values else None


def _hybrid_array_from_parts(fp, descriptor_vector: np.ndarray) -> np.ndarray:
    return np.concatenate([_fingerprint_array(fp).astype(np.float32), descriptor_vector.astype(np.float32)])


def _normalize_model_bundles(models: Any) -> list[ModelBundle]:
    if isinstance(models, dict) and "models" in models:
        models = models["models"]

    bundles: list[ModelBundle] = []
    for idx, item in enumerate(models):
        if isinstance(item, ModelBundle):
            bundles.append(item)
            continue
        if isinstance(item, dict) and "model" in item:
            bundles.append(
                ModelBundle(
                    name=str(item.get("name", f"model_{idx+1}")),
                    feature_set=str(item.get("feature_set", "ecfp")),
                    model=item["model"],
                )
            )
            continue
        bundles.append(ModelBundle(name=f"model_{idx+1}", feature_set="ecfp", model=item))
    return bundles


def _extract_uncertainty_scale(model_artifact: Any) -> float:
    if isinstance(model_artifact, dict) and "uncertainty_scale" in model_artifact:
        try:
            return float(model_artifact["uncertainty_scale"])
        except (TypeError, ValueError):
            return 1.0
    return 1.0


@lru_cache(maxsize=4)
def _load_reference_library(csv_path: str, smiles_col: str, name_col: str | None = None) -> ReferenceLibrary:
    path = Path(csv_path)
    if not path.exists():
        return ReferenceLibrary(smiles=[], fps=[], names=None)

    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        return ReferenceLibrary(smiles=[], fps=[], names=None)

    smiles = []
    fps = []
    names = [] if name_col and name_col in df.columns else None

    for _, row in df.iterrows():
        smi = row.get(smiles_col)
        fp = morgan_fp(smiles=smi)
        if fp is None:
            continue
        smiles.append(smi)
        fps.append(fp)
        if names is not None:
            names.append(str(row.get(name_col, "")))

    return ReferenceLibrary(smiles=smiles, fps=fps, names=names)


def _load_models(model_path: str | None = None):
    if model_path:
        path = Path(model_path)
    else:
        preferred_paths = [
            PROJECT_ROOT / "models" / "qsar_multiview_ensemble.pkl",
            PROJECT_ROOT / "models" / "qsar_rf_ensemble.pkl",
        ]
        path = next((candidate for candidate in preferred_paths if candidate.exists()), preferred_paths[-1])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ensemble model: {path}\n"
            "Run: python -m src.models.train_qsar_rf_ensemble"
        )
    return _normalize_model_bundles(joblib.load(path))


class MultiAgentScorer:
    def __init__(
        self,
        models,
        train_library: ReferenceLibrary,
        marketed_library: ReferenceLibrary | None = None,
        policy: ScoringPolicy | None = None,
        uncertainty_scale: float = 1.0,
    ):
        self.models = _normalize_model_bundles(models)
        self.train_library = train_library
        self.marketed_library = marketed_library or ReferenceLibrary(smiles=[], fps=[], names=None)
        self.policy = policy or ScoringPolicy()
        self.uncertainty_scale = float(max(0.1, uncertainty_scale))

    def _feature_row_for_bundle(self, bundle: ModelBundle, fp, descriptor_vector: np.ndarray) -> np.ndarray:
        if bundle.feature_set == "ecfp":
            return _fingerprint_array(fp).reshape(1, -1)
        if bundle.feature_set == "descriptors":
            return descriptor_vector.reshape(1, -1)
        if bundle.feature_set == "hybrid":
            return _hybrid_array_from_parts(fp, descriptor_vector).reshape(1, -1)
        raise ValueError(f"Unsupported feature_set={bundle.feature_set} for bundle={bundle.name}")

    def _feature_matrix_for_bundle(
        self,
        bundle: ModelBundle,
        fps: list[Any],
        descriptor_matrix: np.ndarray,
    ) -> np.ndarray:
        if bundle.feature_set == "ecfp":
            return np.vstack([_fingerprint_array(fp) for fp in fps])
        if bundle.feature_set == "descriptors":
            return descriptor_matrix
        if bundle.feature_set == "hybrid":
            ecfp_matrix = np.vstack([_fingerprint_array(fp).astype(np.float32) for fp in fps])
            return np.hstack([ecfp_matrix, descriptor_matrix.astype(np.float32)])
        raise ValueError(f"Unsupported feature_set={bundle.feature_set} for bundle={bundle.name}")

    def predict_with_ensemble(self, fp, descriptor_vector: np.ndarray | None = None) -> tuple[float, float]:
        descriptor_row = descriptor_vector if descriptor_vector is not None else np.zeros((13,), dtype=np.float32)
        preds = np.asarray(
            [
                bundle.model.predict(self._feature_row_for_bundle(bundle, fp, descriptor_row))[0]
                for bundle in self.models
            ],
            dtype=float,
        )
        return float(preds.mean()), float(preds.std() * self.uncertainty_scale)

    def predict_many_with_ensemble(self, fps: list[Any], descriptor_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not fps:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        pred_matrix = np.vstack(
            [
                bundle.model.predict(self._feature_matrix_for_bundle(bundle, fps, descriptor_matrix))
                for bundle in self.models
            ]
        ).astype(float)
        return pred_matrix.mean(axis=0), pred_matrix.std(axis=0) * self.uncertainty_scale

    def _score_from_precomputed(
        self,
        canonical_smiles: str,
        mol,
        fp,
        pred_mean: float,
        pred_std: float,
    ) -> dict[str, Any]:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        num_hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        num_hba = int(rdMolDescriptors.CalcNumHBA(mol))
        qed = float(QED.qed(mol))
        sa = float(simple_sa_score(canonical_smiles) or 10.0)
        ring_count = int(rdMolDescriptors.CalcNumRings(mol))
        rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))

        lipinski_violations = int(mw > 500) + int(logp > 5) + int(num_hbd > 5) + int(num_hba > 10)
        property_penalty = (
            0.50 * int(mw > 500)
            + 0.45 * int(logp > 5)
            + 0.25 * int(tpsa > 140)
            + 0.20 * int(num_hbd > 5)
            + 0.20 * int(num_hba > 10)
        )

        has_pains, pains_desc = pains_alert(canonical_smiles)
        alert_names = structural_alerts(canonical_smiles)
        severe_alert_names = severe_structural_alerts(canonical_smiles)
        warhead_alert_names = covalent_warhead_alerts(canonical_smiles)

        train_sims = bulk_tanimoto_similarity(fp, self.train_library.fps)
        max_train_similarity = max(train_sims) if train_sims else 0.0
        top5_train_similarity = top_k_mean(train_sims, k=5)

        market_sims = bulk_tanimoto_similarity(fp, self.marketed_library.fps)
        max_market_similarity = max(market_sims) if market_sims else 0.0
        closest_market_name = None
        if market_sims and self.marketed_library.names:
            closest_idx = int(np.argmax(market_sims))
            closest_market_name = self.marketed_library.names[closest_idx]

        applicability_score = min(1.0, max(0.0, (top5_train_similarity - 0.15) / 0.55))
        novelty_score = max(0.0, 1.0 - max_train_similarity)
        market_novelty_score = max(0.0, 1.0 - max_market_similarity)

        potency_reward = pred_mean - self.policy.potency_uncertainty_weight * pred_std
        chemistry_reward = (
            1.00 * qed
            - 0.15 * sa
            - 0.25 * lipinski_violations
            - property_penalty
            + 0.10 * fraction_csp3
        )
        safety_reward = (
            -0.40 * int(has_pains)
            - 0.20 * len(alert_names)
            - 0.30 * len(severe_alert_names)
        )
        novelty_reward = (
            0.45 * novelty_score
            + 0.30 * market_novelty_score
            + 0.25 * applicability_score
        )

        if max_train_similarity < self.policy.extreme_novelty_similarity_threshold:
            novelty_reward -= 0.75
        if max_train_similarity > 0.96:
            novelty_reward -= 0.20

        naive_reward = potency_reward + chemistry_reward + safety_reward + novelty_reward

        potency_support = _clamp01(((pred_mean - 5.5) / 4.0) - 0.70 * pred_std)
        chemistry_support = _clamp01(
            0.55 * qed
            + 0.25 * (1.0 - min(sa / self.policy.sa_veto_threshold, 1.0))
            + 0.20 * (1.0 - min(lipinski_violations / 4.0, 1.0))
        )
        safety_support = _clamp01(
            1.0
            - 0.25 * int(has_pains)
            - 0.12 * len(alert_names)
            - 0.18 * len(severe_alert_names)
            - 0.08 * len(warhead_alert_names)
        )
        domain_support = _clamp01(0.75 * applicability_score + 0.25 * market_novelty_score)
        support_values = np.asarray(
            [potency_support, chemistry_support, safety_support, domain_support],
            dtype=float,
        )
        agent_support_min = float(np.min(support_values))
        agent_support_mean = float(np.mean(support_values))
        agent_support_max = float(np.max(support_values))
        agent_disagreement_score = float(agent_support_max - agent_support_min)
        multi_agent_balance = _clamp01(1.0 - agent_disagreement_score)
        potency_vs_guardrails_gap = float(
            potency_support - np.mean([chemistry_support, safety_support, domain_support])
        )

        veto_reasons = []
        if qed < self.policy.qed_veto_threshold:
            veto_reasons.append("very_low_qed")
        if sa > self.policy.sa_veto_threshold:
            veto_reasons.append("poor_synthetic_accessibility")
        if lipinski_violations >= 3:
            veto_reasons.append("too_many_lipinski_violations")
        if len(severe_alert_names) >= 2:
            veto_reasons.append("multiple_severe_structural_alerts")
        if pred_std > self.policy.uncertainty_veto_threshold:
            veto_reasons.append("high_model_uncertainty")
        if pred_mean >= 9.5 and max_train_similarity < 0.10:
            veto_reasons.append("extreme_out_of_domain_potency")

        reward_hacking_flags = []
        reward_hacking_risk = 0.0

        if pred_mean >= self.policy.suspicious_potency_threshold and max_train_similarity < self.policy.out_of_domain_similarity_threshold:
            reward_hacking_flags.append("potent_but_outside_applicability_domain")
            reward_hacking_risk += 0.35
        if pred_mean >= self.policy.suspicious_potency_threshold and (qed < 0.35 or sa > 4.5):
            reward_hacking_flags.append("potency_dominates_drug_likeness")
            reward_hacking_risk += 0.20
        if len(alert_names) >= 2 or has_pains:
            reward_hacking_flags.append("unsafe_structure_can_game_single_score")
            reward_hacking_risk += 0.25
        if pred_std > self.policy.suspicious_uncertainty_threshold:
            reward_hacking_flags.append("uncertain_prediction")
            reward_hacking_risk += 0.15
        if max_market_similarity > self.policy.near_copy_similarity_threshold:
            reward_hacking_flags.append("near_copy_of_marketed_scaffold")
            reward_hacking_risk += 0.10
        if applicability_score < self.policy.audit_pass_min_applicability and novelty_score > 0.75:
            reward_hacking_flags.append("novelty_without_domain_support")
            reward_hacking_risk += 0.15
        if agent_disagreement_score > self.policy.audit_pass_disagreement_threshold:
            reward_hacking_flags.append("agents_strongly_disagree")
            reward_hacking_risk += 0.20
        if potency_support > 0.85 and agent_support_min < 0.35:
            reward_hacking_flags.append("potency_not_supported_by_other_agents")
            reward_hacking_risk += 0.20
        if warhead_alert_names and pred_mean >= 8.7:
            reward_hacking_flags.append("reactive_warhead_could_inflate_proxy_score")
            reward_hacking_risk += 0.15

        reward_hacking_risk = min(1.0, reward_hacking_risk)
        risk_level = "low"
        if reward_hacking_risk >= 0.75:
            risk_level = "critical"
        elif reward_hacking_risk >= 0.50:
            risk_level = "high"
        elif reward_hacking_risk >= 0.25:
            risk_level = "medium"

        applicability_penalty = max(
            0.0,
            self.policy.audit_pass_min_applicability - applicability_score,
        ) * 2.0
        market_copy_penalty = max(
            0.0,
            max_market_similarity - self.policy.marketed_copy_penalty_threshold,
        ) * 2.5
        uncertainty_penalty = max(0.0, pred_std - 0.10) * 1.5
        disagreement_penalty = max(0.0, agent_disagreement_score - 0.30) * 1.25
        reactivity_penalty = 0.10 * len(warhead_alert_names)
        anti_hacking_penalty = (
            applicability_penalty
            + market_copy_penalty
            + uncertainty_penalty
            + disagreement_penalty
            + reactivity_penalty
            + 0.75 * reward_hacking_risk
        )
        verified_reward = naive_reward - anti_hacking_penalty

        audit_pass = (
            not veto_reasons
            and reward_hacking_risk < self.policy.audit_pass_risk_threshold
            and agent_disagreement_score <= self.policy.audit_pass_disagreement_threshold
            and applicability_score >= self.policy.audit_pass_min_applicability
        )
        audit_status = "pass"
        if veto_reasons or reward_hacking_risk >= 0.65 or applicability_score < 0.15:
            audit_status = "fail"
        elif not audit_pass:
            audit_status = "review"

        return {
            "smiles": canonical_smiles,
            "predicted_pIC50": pred_mean,
            "uncertainty": pred_std,
            "QED": qed,
            "MW": mw,
            "LogP": logp,
            "TPSA": tpsa,
            "HBD": num_hbd,
            "HBA": num_hba,
            "ring_count": ring_count,
            "rotatable_bonds": rot_bonds,
            "fraction_csp3": fraction_csp3,
            "lipinski_violations": lipinski_violations,
            "penalty": property_penalty,
            "SA_score": sa,
            "has_PAINS": has_pains,
            "PAINS_alert": pains_desc,
            "structural_alert_count": len(alert_names),
            "structural_alerts": ";".join(alert_names) if alert_names else None,
            "severe_alert_count": len(severe_alert_names),
            "severe_alerts": ";".join(severe_alert_names) if severe_alert_names else None,
            "covalent_warhead_count": len(warhead_alert_names),
            "covalent_warheads": ";".join(warhead_alert_names) if warhead_alert_names else None,
            "max_train_similarity": max_train_similarity,
            "top5_train_similarity": top5_train_similarity,
            "applicability_score": applicability_score,
            "max_market_similarity": max_market_similarity,
            "closest_market_name": closest_market_name,
            "novelty_score": novelty_score,
            "market_novelty_score": market_novelty_score,
            "potency_support": potency_support,
            "chemistry_support": chemistry_support,
            "safety_support": safety_support,
            "domain_support": domain_support,
            "agent_support_min": agent_support_min,
            "agent_support_mean": agent_support_mean,
            "agent_support_max": agent_support_max,
            "agent_disagreement_score": agent_disagreement_score,
            "multi_agent_balance": multi_agent_balance,
            "potency_vs_guardrails_gap": potency_vs_guardrails_gap,
            "potency_reward": potency_reward,
            "chemistry_reward": chemistry_reward,
            "safety_reward": safety_reward,
            "novelty_reward": novelty_reward,
            "naive_reward": naive_reward,
            "applicability_penalty": applicability_penalty,
            "market_copy_penalty": market_copy_penalty,
            "uncertainty_penalty": uncertainty_penalty,
            "disagreement_penalty": disagreement_penalty,
            "reactivity_penalty": reactivity_penalty,
            "anti_hacking_penalty": anti_hacking_penalty,
            "reward_hacking_risk": reward_hacking_risk,
            "reward_hacking_risk_level": risk_level,
            "reward_hacking_flags": _safe_join(reward_hacking_flags),
            "verified_reward": verified_reward,
            "audit_pass": audit_pass,
            "audit_status": audit_status,
            "veto": bool(veto_reasons),
            "veto_reasons": _safe_join(veto_reasons),
        }

    def score(self, smiles: str) -> dict[str, Any]:
        mol = mol_from_smiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        fp = morgan_fp(mol=mol)
        descriptor_vector = descriptor_vector_from_mol(mol)
        pred_mean, pred_std = self.predict_with_ensemble(fp, descriptor_vector=descriptor_vector)
        return self._score_from_precomputed(canonical_smiles, mol, fp, pred_mean, pred_std)


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    filled = series.fillna(series.median() if not series.dropna().empty else 0.0)
    return filled.rank(method="average", pct=True, ascending=higher_is_better)


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def resolve_priority_score_column(df: pd.DataFrame) -> str:
    for column in [
        "prospective_acquisition_score",
        "experimental_readiness_priority",
        "feasible_priority_score",
        "interaction_priority_score",
        "structural_priority_score",
        "final_score",
    ]:
        if column in df.columns:
            return column
    raise ValueError("Unable to resolve a priority score column for structural ranking.")


def add_multiobjective_ranking(df: pd.DataFrame, policy: ScoringPolicy | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    active_policy = policy or ScoringPolicy()
    out = df.copy()
    out["potency_percentile"] = _percentile_rank(out["predicted_pIC50"], higher_is_better=True)
    out["qed_percentile"] = _percentile_rank(out["QED"], higher_is_better=True)
    out["sa_percentile"] = _percentile_rank(out["SA_score"], higher_is_better=False)
    out["novelty_percentile"] = _percentile_rank(out["novelty_score"], higher_is_better=True)
    out["applicability_percentile"] = _percentile_rank(out["applicability_score"], higher_is_better=True)
    out["balance_percentile"] = _percentile_rank(out["multi_agent_balance"], higher_is_better=True)
    out["risk_percentile"] = _percentile_rank(out["reward_hacking_risk"], higher_is_better=False)
    out["uncertainty_percentile"] = _percentile_rank(out["uncertainty"], higher_is_better=False)

    out["naive_multi_objective_score"] = (
        0.45 * out["potency_percentile"]
        + 0.20 * out["qed_percentile"]
        + 0.15 * out["sa_percentile"]
        + 0.20 * out["novelty_percentile"]
    )
    out["multi_objective_score"] = (
        0.30 * out["potency_percentile"]
        + 0.15 * out["qed_percentile"]
        + 0.10 * out["sa_percentile"]
        + 0.10 * out["novelty_percentile"]
        + 0.10 * out["applicability_percentile"]
        + 0.10 * out["balance_percentile"]
        + 0.10 * out["risk_percentile"]
        + 0.05 * out["uncertainty_percentile"]
    )
    out["naive_score"] = out["naive_reward"] + active_policy.multiobjective_weight * out["naive_multi_objective_score"]
    out["naive_rank"] = out["naive_score"].rank(method="first", ascending=False).astype(int)

    status_penalty = out["audit_status"].map({"pass": 0.0, "review": 0.75, "fail": 1.50}).fillna(0.75)
    out["audit_status_penalty"] = status_penalty

    out["final_score"] = (
        out["verified_reward"]
        + active_policy.multiobjective_weight * out["multi_objective_score"]
        - active_policy.final_risk_weight * out["reward_hacking_risk"]
        - active_policy.final_veto_weight * out["veto"].astype(float)
        - out["audit_status_penalty"]
    )

    out["audit_priority"] = out["audit_status"].map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    out = out.sort_values(
        ["veto", "audit_priority", "final_score", "predicted_pIC50", "QED"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["audit_demote_positions"] = (out["rank"] - out["naive_rank"]).clip(lower=0)
    out["audit_promote_positions"] = (out["naive_rank"] - out["rank"]).clip(lower=0)
    out["selection_bucket"] = np.select(
        [
            out["veto"] | (out["audit_status"] == "fail"),
            (out["audit_status"] == "review") | (out["reward_hacking_risk"] >= 0.25),
            out["rank"] <= 25,
        ],
        ["reject", "review", "lead"],
        default="advance",
    )
    return out


def add_structure_agent_ranking(
    df: pd.DataFrame,
    policy: ScoringPolicy | None = None,
    base_score_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    active_policy = policy or ScoringPolicy()
    out = df.copy()
    base_column = base_score_col or resolve_priority_score_column(out)

    docking_support = _series_or_default(out, "docking_rescore", 0.0).clip(lower=0.0, upper=1.0)
    interaction_support = _series_or_default(out, "interaction_support_score", 0.0).clip(lower=0.0, upper=1.0)
    key_residue_support = (_series_or_default(out, "interaction_key_residue_count", 0.0) / 4.0).clip(lower=0.0, upper=1.0)
    vina_support = ((-_series_or_default(out, "vina_affinity_kcal", -6.0) - 6.5) / 2.5).clip(lower=0.0, upper=1.0)
    feasibility_support = _series_or_default(out, "feasibility_score", 0.0).clip(lower=0.0, upper=1.0)
    risk_support = (1.0 - _series_or_default(out, "reward_hacking_risk", 0.5)).clip(lower=0.0, upper=1.0)

    uncertainty_series = _series_or_default(out, "uncertainty", 0.20)
    if uncertainty_series.empty:
        uncertainty_support = pd.Series(0.5, index=out.index, dtype=float)
    else:
        uncertainty_scale = max(0.10, float(uncertainty_series.quantile(0.90)))
        uncertainty_support = (1.0 - (uncertainty_series / uncertainty_scale)).clip(lower=0.0, upper=1.0)

    out["structure_docking_support"] = docking_support
    out["structure_interaction_support"] = interaction_support
    out["structure_key_residue_support"] = key_residue_support
    out["structure_vina_support"] = vina_support
    out["structure_guardrail_support"] = (
        0.50 * feasibility_support
        + 0.30 * risk_support
        + 0.20 * uncertainty_support
    ).clip(lower=0.0, upper=1.0)
    out["structure_agent_support"] = (
        0.32 * docking_support
        + 0.28 * interaction_support
        + 0.16 * key_residue_support
        + 0.14 * vina_support
        + 0.10 * feasibility_support
    ).clip(lower=0.0, upper=1.0)

    if "closest_market_docking_rescore" in out.columns:
        out["structure_vs_market_gap"] = docking_support - _series_or_default(out, "closest_market_docking_rescore", 0.0)
    else:
        out["structure_vs_market_gap"] = docking_support - 0.50

    out["structure_percentile"] = _percentile_rank(out["structure_agent_support"], higher_is_better=True)
    out["interaction_percentile"] = _percentile_rank(out["structure_interaction_support"], higher_is_better=True)
    out["structure_guardrail_percentile"] = _percentile_rank(out["structure_guardrail_support"], higher_is_better=True)
    out["structure_market_gap_percentile"] = _percentile_rank(out["structure_vs_market_gap"], higher_is_better=True)
    out["structure_base_percentile"] = _percentile_rank(_series_or_default(out, base_column, 0.0), higher_is_better=True)

    out["structure_augmented_score"] = (
        _series_or_default(out, base_column, 0.0)
        + 0.90 * out["structure_agent_support"]
        + 0.40 * out["structure_guardrail_support"]
        + 0.30 * out["structure_percentile"]
        + 0.20 * out["interaction_percentile"]
        + 0.10 * out["structure_market_gap_percentile"]
    )
    out["structure_agent_disagreement"] = (
        out[["structure_docking_support", "structure_interaction_support", "structure_guardrail_support"]].max(axis=1)
        - out[["structure_docking_support", "structure_interaction_support", "structure_guardrail_support"]].min(axis=1)
    )
    out["structure_agent_status"] = np.select(
        [
            (_series_or_default(out, "veto", 0.0) >= 1.0)
            | (_series_or_default(out, "reward_hacking_risk", 0.0) >= 0.60)
            | (out["structure_guardrail_support"] < 0.35),
            (out["structure_agent_support"] < 0.40)
            | (out["structure_interaction_support"] < 0.30)
            | (out["structure_agent_disagreement"] > active_policy.audit_pass_disagreement_threshold),
        ],
        ["fail", "review"],
        default="pass",
    )

    out["audit_priority"] = out.get("audit_status", pd.Series("review", index=out.index)).map(
        {"pass": 0, "review": 1, "fail": 2}
    ).fillna(1).astype(int)
    out["structure_priority"] = out["structure_agent_status"].map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)

    out = out.sort_values(
        [
            "veto" if "veto" in out.columns else "structure_priority",
            "audit_priority",
            "structure_priority",
            "structure_augmented_score",
            base_column,
            "predicted_pIC50" if "predicted_pIC50" in out.columns else "structure_percentile",
        ],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)
    out["structure_rank"] = np.arange(1, len(out) + 1)
    return out


def build_default_scorer(models=None, policy: ScoringPolicy | None = None) -> MultiAgentScorer:
    model_artifact = models or _load_models()
    dataset_path = resolve_preferred_processed_dataset()
    train_library = _load_reference_library(
        str(dataset_path),
        smiles_col="smiles_canonical",
    )
    marketed_library = _load_reference_library(
        str(PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"),
        smiles_col="smiles",
        name_col="name",
    )
    return MultiAgentScorer(
        models=model_artifact,
        train_library=train_library,
        marketed_library=marketed_library,
        policy=policy,
        uncertainty_scale=_extract_uncertainty_scale(model_artifact),
    )


def score_smiles_list(smiles_list: Iterable[str], scorer: MultiAgentScorer) -> pd.DataFrame:
    prepared = []
    seen_smiles: set[str] = set()

    for smiles in smiles_list:
        try:
            mol = mol_from_smiles(smiles)
            if mol is None:
                continue
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if canonical_smiles in seen_smiles:
                continue
            fp = morgan_fp(mol=mol)
            if fp is None:
                continue
            descriptor_vector = descriptor_vector_from_mol(mol)
            prepared.append((canonical_smiles, mol, fp, descriptor_vector))
            seen_smiles.add(canonical_smiles)
        except Exception:
            continue

    if not prepared:
        return pd.DataFrame()

    pred_means, pred_stds = scorer.predict_many_with_ensemble(
        [fp for _, _, fp, _ in prepared],
        np.vstack([descriptor_vector for _, _, _, descriptor_vector in prepared]).astype(np.float32),
    )
    rows = [
        scorer._score_from_precomputed(canonical_smiles, mol, fp, float(pred_mean), float(pred_std))
        for (canonical_smiles, mol, fp, _), pred_mean, pred_std in zip(prepared, pred_means, pred_stds)
    ]

    out = pd.DataFrame(rows).reset_index(drop=True)
    return add_multiobjective_ranking(out, policy=scorer.policy)
