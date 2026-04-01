from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerifiableRewardWeights:
    delta_verified_reward: float = 1.00
    delta_potency: float = 0.55
    delta_qed: float = 0.20
    feasibility_score: float = 0.80
    feasibility_pass_bonus: float = 0.25
    feasibility_fail_penalty: float = 0.60
    feasibility_hard_gate_fail_penalty: float = 0.80
    structural_support_bonus: float = 0.20
    interaction_support_bonus: float = 0.25
    medchem_realism_bonus: float = 0.18
    synthetic_route_bonus: float = 0.16
    generator_priority_bonus: float = 0.18
    adaptive_action_prior_bonus: float = 0.14
    parent_similarity_bonus: float = 0.14
    scaffold_preservation_bonus: float = 0.10
    audit_pass_bonus: float = 0.20
    audit_review_penalty: float = 0.15
    audit_fail_penalty: float = 0.50
    reward_hacking_penalty: float = 0.80
    veto_penalty: float = 0.60
    stagnation_penalty: float = 0.10


def compute_verifiable_reward(
    parent_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
    feasibility_profile: dict[str, Any],
    weights: VerifiableRewardWeights | None = None,
) -> dict[str, float | str]:
    active = weights or VerifiableRewardWeights()
    delta_verified = float(candidate_profile["verified_reward"]) - float(parent_profile["verified_reward"])
    delta_potency = float(candidate_profile["predicted_pIC50"]) - float(parent_profile["predicted_pIC50"])
    delta_qed = float(candidate_profile["QED"]) - float(parent_profile["QED"])

    feasibility_score = float(feasibility_profile["feasibility_score"])
    feasibility_status = str(feasibility_profile["feasibility_status"])
    feasibility_hard_gate_pass = bool(feasibility_profile.get("feasibility_hard_gate_pass", True))
    structural_support_score = float(feasibility_profile.get("structural_support_score", 0.0))
    interaction_support_score = float(feasibility_profile.get("interaction_support_score", 0.0))
    medchem_realism_score = float(feasibility_profile.get("medchem_realism_score", 0.0))
    synthetic_route_score = float(feasibility_profile.get("route_synthetic_support_score", 0.0))
    generator_priority_score = float(candidate_profile.get("generator_priority_score", 0.0))
    adaptive_action_prior = float(candidate_profile.get("adaptive_action_prior", 0.50))
    parent_similarity = float(candidate_profile.get("parent_similarity", feasibility_profile.get("parent_similarity", 0.0)) or 0.0)
    preserves_scaffold = bool(candidate_profile.get("preserves_scaffold", True))
    audit_status = str(candidate_profile["audit_status"])
    hacking_risk = float(candidate_profile["reward_hacking_risk"])
    veto = bool(candidate_profile["veto"])

    reward = (
        active.delta_verified_reward * delta_verified
        + active.delta_potency * delta_potency
        + active.delta_qed * delta_qed
        + active.feasibility_score * feasibility_score
        + active.feasibility_pass_bonus * float(feasibility_status == "pass")
        + active.structural_support_bonus * structural_support_score
        + active.interaction_support_bonus * interaction_support_score
        + active.medchem_realism_bonus * medchem_realism_score
        + active.synthetic_route_bonus * synthetic_route_score
        + active.generator_priority_bonus * generator_priority_score
        + active.adaptive_action_prior_bonus * adaptive_action_prior
        + active.parent_similarity_bonus * parent_similarity
        + active.scaffold_preservation_bonus * float(preserves_scaffold)
        - active.feasibility_fail_penalty * float(feasibility_status == "fail")
        - active.feasibility_hard_gate_fail_penalty * float(not feasibility_hard_gate_pass)
        + active.audit_pass_bonus * float(audit_status == "pass")
        - active.audit_review_penalty * float(audit_status == "review")
        - active.audit_fail_penalty * float(audit_status == "fail")
        - active.reward_hacking_penalty * hacking_risk
        - active.veto_penalty * float(veto)
    )

    if delta_verified <= 0 and delta_potency <= 0:
        reward -= active.stagnation_penalty

    return {
        "reward_total": float(reward),
        "reward_delta_verified": float(delta_verified),
        "reward_delta_potency": float(delta_potency),
        "reward_delta_qed": float(delta_qed),
        "reward_feasibility": feasibility_score,
        "reward_structural_support": structural_support_score,
        "reward_interaction_support": interaction_support_score,
        "reward_medchem_realism": medchem_realism_score,
        "reward_synthetic_route": synthetic_route_score,
        "reward_generator_priority": generator_priority_score,
        "reward_adaptive_action_prior": adaptive_action_prior,
        "reward_parent_similarity": parent_similarity,
        "reward_feasibility_hard_gate": float(-active.feasibility_hard_gate_fail_penalty * float(not feasibility_hard_gate_pass)),
        "reward_hacking_penalty": float(-active.reward_hacking_penalty * hacking_risk),
        "reward_veto_penalty": float(-active.veto_penalty * float(veto)),
        "reward_audit_status": audit_status,
        "reward_feasibility_status": feasibility_status,
    }
