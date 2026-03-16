from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.agents.multi_agent import MultiAgentScorer, build_default_scorer, score_smiles_list
from src.feasibility.assessor import FeasibilityAssessor
from src.generation.medchem_mutations import MutationOutcome, generate_medchem_outcomes
from src.rl.rewarding import VerifiableRewardWeights, compute_verifiable_reward


@dataclass(frozen=True)
class GroundedAction:
    action_id: str
    action_name: str
    action_category: str
    candidate_smiles: str
    candidate_profile: dict[str, Any]
    feasibility_profile: dict[str, Any]
    reward_profile: dict[str, Any]
    selection_rank: int = 1


class VerifiableMoleculeEnv:
    def __init__(
        self,
        seed_smiles: list[str],
        scorer: MultiAgentScorer | None = None,
        assessor: FeasibilityAssessor | None = None,
        reward_weights: VerifiableRewardWeights | None = None,
        max_steps: int = 4,
        max_variants_per_step: int = 60,
        max_actions_per_family: int = 2,
        max_actions_total: int = 18,
        random_seed: int = 42,
    ):
        if not seed_smiles:
            raise ValueError("VerifiableMoleculeEnv requires at least one seed molecule.")
        self.seed_smiles = list(seed_smiles)
        self.scorer = scorer or build_default_scorer()
        self.assessor = assessor or FeasibilityAssessor()
        self.reward_weights = reward_weights or VerifiableRewardWeights()
        self.max_steps = int(max_steps)
        self.max_variants_per_step = int(max_variants_per_step)
        self.max_actions_per_family = int(max(1, max_actions_per_family))
        self.max_actions_total = int(max(2, max_actions_total))
        self.random = random.Random(random_seed)

        self.current_smiles: str | None = None
        self.current_profile: dict[str, Any] | None = None
        self.current_feasibility: dict[str, Any] | None = None
        self.step_idx: int = 0
        self.ledger: list[dict[str, Any]] = []
        self._cached_actions: list[GroundedAction] | None = None

    def _score_single(self, smiles: str) -> dict[str, Any]:
        return self.scorer.score(smiles)

    def _state_key(self) -> str:
        if self.current_profile is None:
            return "uninitialized"
        potency = float(self.current_profile["predicted_pIC50"])
        risk = float(self.current_profile["reward_hacking_risk"])
        audit = str(self.current_profile["audit_status"])
        feasibility_score = float((self.current_feasibility or {}).get("feasibility_score", 0.50))
        potency_bucket = "low" if potency < 8.0 else ("mid" if potency < 9.0 else "high")
        risk_bucket = "low" if risk < 0.10 else ("mid" if risk < 0.30 else "high")
        feasibility_bucket = "low" if feasibility_score < 0.55 else ("mid" if feasibility_score < 0.75 else "high")
        return f"step={self.step_idx}|potency={potency_bucket}|risk={risk_bucket}|feasibility={feasibility_bucket}|audit={audit}"

    def reset(self, seed_smiles: str | None = None) -> str:
        self.current_smiles = seed_smiles or self.random.choice(self.seed_smiles)
        self.current_profile = self._score_single(self.current_smiles)
        self.current_feasibility = self.assessor.assess(self.current_smiles)
        self.step_idx = 0
        self.ledger = []
        self._cached_actions = None
        return self._state_key()

    def _ground_actions(self) -> list[GroundedAction]:
        if self.current_smiles is None or self.current_profile is None:
            raise RuntimeError("Environment must be reset before requesting actions.")
        if self._cached_actions is not None:
            return self._cached_actions

        outcomes = generate_medchem_outcomes(self.current_smiles, max_variants=self.max_variants_per_step)
        if not outcomes:
            self._cached_actions = []
            return self._cached_actions

        outcome_map: dict[str, MutationOutcome] = {}
        for outcome in outcomes:
            outcome_map[outcome.smiles] = outcome

        scored = score_smiles_list(list(outcome_map.keys()), scorer=self.scorer)
        if scored.empty:
            self._cached_actions = []
            return self._cached_actions

        grouped: dict[str, list[GroundedAction]] = {}
        for _, row in scored.iterrows():
            smiles = str(row["smiles"])
            outcome = outcome_map.get(smiles)
            if outcome is None:
                continue
            candidate_profile = row.to_dict()
            candidate_profile.update(
                {
                    "action_rule_source": outcome.rule_source,
                    "reaction_family": outcome.reaction_family,
                    "synthetic_route": outcome.synthetic_route,
                    "synthetic_feasibility_score": outcome.synthetic_feasibility_score,
                    "medchem_realism_score": outcome.medchem_realism_score,
                    "transformation_confidence_score": outcome.transformation_confidence,
                    "preserves_scaffold": outcome.preserves_scaffold,
                    "parent_similarity": outcome.parent_similarity,
                    "property_support_score": outcome.property_support_score,
                    "category_priority_score": outcome.category_priority_score,
                    "generator_priority_score": outcome.generator_priority_score,
                    "hard_constraint_pass": outcome.hard_constraint_pass,
                    "hard_constraint_notes": outcome.hard_constraint_notes,
                    "introduced_warhead": outcome.introduced_warhead,
                    "warhead_retained": outcome.warhead_retained,
                    "alert_count": outcome.alert_count,
                    "severe_alert_count": outcome.severe_alert_count,
                }
            )
            feasibility_profile = self.assessor.assess(
                smiles,
                parent_smiles=self.current_smiles,
                action_name=outcome.action_name,
                synthetic_feasibility_score=outcome.synthetic_feasibility_score,
                medchem_realism_score=outcome.medchem_realism_score,
                transformation_confidence=outcome.transformation_confidence,
                reaction_family=outcome.reaction_family,
            )
            reward_profile = compute_verifiable_reward(
                self.current_profile,
                candidate_profile,
                feasibility_profile,
                weights=self.reward_weights,
            )
            grounded = GroundedAction(
                action_id=outcome.action_name,
                action_name=outcome.action_name,
                action_category=outcome.category,
                candidate_smiles=smiles,
                candidate_profile=candidate_profile,
                feasibility_profile=feasibility_profile,
                reward_profile=reward_profile,
            )
            grouped.setdefault(outcome.action_name, []).append(grounded)

        best_actions: list[GroundedAction] = []
        for action_name, items in grouped.items():
            ranked_items = sorted(
                items,
                key=lambda item: (
                    float(item.reward_profile["reward_total"]),
                    float(item.candidate_profile.get("generator_priority_score", 0.0)),
                    float(item.candidate_profile.get("final_score", 0.0)),
                    float(item.feasibility_profile.get("feasibility_score", 0.0)),
                ),
                reverse=True,
            )
            for selection_rank, item in enumerate(ranked_items[: self.max_actions_per_family], start=1):
                best_actions.append(
                    GroundedAction(
                        action_id=f"{action_name}__{selection_rank}",
                        action_name=item.action_name,
                        action_category=item.action_category,
                        candidate_smiles=item.candidate_smiles,
                        candidate_profile=item.candidate_profile,
                        feasibility_profile=item.feasibility_profile,
                        reward_profile=item.reward_profile,
                        selection_rank=selection_rank,
                    )
                )
        best_actions.sort(
            key=lambda item: (
                float(item.reward_profile["reward_total"]),
                float(item.candidate_profile.get("generator_priority_score", 0.0)),
                float(item.candidate_profile.get("final_score", 0.0)),
            ),
            reverse=True,
        )
        self._cached_actions = best_actions[: self.max_actions_total]
        return self._cached_actions

    def available_actions(self) -> list[GroundedAction]:
        return self._ground_actions()

    def step(self, action_key: str) -> tuple[str, float, bool, dict[str, Any]]:
        actions = self._ground_actions()
        chosen = next(
            (
                action
                for action in actions
                if action.action_id == action_key or action.action_name == action_key
            ),
            None,
        )
        if chosen is None:
            raise ValueError(f"Unknown or unavailable action: {action_key}")

        self.step_idx += 1
        self.current_smiles = chosen.candidate_smiles
        self.current_profile = chosen.candidate_profile
        self.current_feasibility = chosen.feasibility_profile
        self._cached_actions = None

        ledger_entry = {
            "step": self.step_idx,
            "smiles": chosen.candidate_smiles,
            "action_id": chosen.action_id,
            "action_name": chosen.action_name,
            "action_category": chosen.action_category,
            "action_selection_rank": chosen.selection_rank,
            **chosen.candidate_profile,
            **chosen.feasibility_profile,
            **chosen.reward_profile,
        }
        self.ledger.append(ledger_entry)

        done = self.step_idx >= self.max_steps or not self._ground_actions()
        return self._state_key(), float(chosen.reward_profile["reward_total"]), done, ledger_entry

    def current_observation(self) -> dict[str, Any]:
        return {
            "smiles": self.current_smiles,
            "step": self.step_idx,
            "state_key": self._state_key(),
            **(self.current_profile or {}),
            **(self.current_feasibility or {}),
        }

    def ledger_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.ledger)
