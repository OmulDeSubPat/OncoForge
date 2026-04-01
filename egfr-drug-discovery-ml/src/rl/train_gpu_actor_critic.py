from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import PROJECT_ROOT
from src.feasibility.assessor import FeasibilityAssessor
from src.rl.environment import GroundedAction, VerifiableMoleculeEnv
from src.rl.train_gpu_dqn import (
    _action_features,
    _plot_curve,
    _postprocess,
    _resolve_device,
    _select_seed_pool,
    _series,
)


@dataclass(frozen=True)
class EpisodeStep:
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    entropy: torch.Tensor


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.08),
            nn.Linear(256, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )
        self.policy_head = nn.Linear(160, 1)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )

    def policy_logits(self, action_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(action_features)
        return self.policy_head(encoded).squeeze(-1)

    def value(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.value_head(state_features).squeeze(-1)


class ActorCriticAgent:
    def __init__(self, input_dim: int, device: torch.device):
        self.device = device
        self.network = PolicyValueNet(input_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=6e-4, weight_decay=1e-4)
        self.gamma = 0.95
        self.entropy_weight = 0.015
        self.value_weight = 0.55
        self.random_exploration = 0.06

    def choose(
        self,
        env: VerifiableMoleculeEnv,
        actions: list[GroundedAction],
        *,
        greedy: bool = False,
    ) -> tuple[GroundedAction, np.ndarray, torch.Tensor, torch.Tensor]:
        action_matrix = np.vstack([_action_features(env, action) for action in actions]).astype(np.float32)
        action_tensor = torch.from_numpy(action_matrix).to(self.device)
        state_vector = torch.from_numpy(action_matrix.mean(axis=0, keepdims=True)).to(self.device)

        logits = self.network.policy_logits(action_tensor)
        priors = torch.tensor(
            [
                float(action.reward_profile.get("reward_total", 0.0))
                + 0.14 * float(action.candidate_profile.get("generator_priority_score", 0.0))
                + 0.10 * float(action.candidate_profile.get("structural_guidance_score", 0.0))
                + 0.08 * float(action.candidate_profile.get("adaptive_action_prior", 0.5))
                for action in actions
            ],
            dtype=torch.float32,
            device=self.device,
        )
        logits = logits + 0.10 * priors
        log_probs_all = torch.log_softmax(logits, dim=0)
        probs = torch.softmax(logits, dim=0)
        value = self.network.value(state_vector).squeeze(0)
        if greedy:
            idx = int(torch.argmax(logits).item())
        else:
            if torch.rand(1, device=self.device).item() < self.random_exploration:
                idx = int(torch.randint(0, len(actions), (1,), device=self.device).item())
            else:
                idx = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = log_probs_all[idx]
        entropy = -(probs * log_probs_all).sum()
        return actions[idx], action_matrix[idx], log_prob, entropy + 0.0 * value

    def update(self, trajectory: list[EpisodeStep]) -> float:
        if not trajectory:
            return 0.0
        returns: list[float] = []
        running = 0.0
        for step in reversed(trajectory):
            running = float(step.reward) + self.gamma * running
            returns.append(running)
        returns = list(reversed(returns))
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack([step.value for step in trajectory]).to(self.device)
        log_probs = torch.stack([step.log_prob for step in trajectory]).to(self.device)
        entropies = torch.stack([step.entropy for step in trajectory]).to(self.device)
        advantages = returns_tensor - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.nn.functional.smooth_l1_loss(values, returns_tensor)
        entropy_bonus = entropies.mean()
        loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy_bonus

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=4.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def save(self, path) -> None:
        torch.save(self.network.state_dict(), path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a GPU actor-critic policy over verifiable medicinal-chemistry actions.")
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed-pool-size", type=int, default=32)
    parser.add_argument("--evaluation-rollouts", type=int, default=16)
    parser.add_argument("--structural-top-k", type=int, default=16)
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-actions-per-family", type=int, default=4)
    parser.add_argument("--max-actions-total", type=int, default=30)
    parser.add_argument("--structure-guidance-budget", type=int, default=32)
    args = parser.parse_args(argv)

    device, device_label = _resolve_device(prefer_gpu=not args.cpu_only)
    seeds = _select_seed_pool(pool_size=args.seed_pool_size)
    env = VerifiableMoleculeEnv(
        seeds,
        assessor=FeasibilityAssessor(),
        max_steps=args.max_steps,
        max_actions_per_family=args.max_actions_per_family,
        max_actions_total=args.max_actions_total,
        structure_guidance_budget=args.structure_guidance_budget,
    )
    env.reset()
    sample_actions = env.available_actions()
    if not sample_actions:
        raise RuntimeError("No RL actions available from the selected seed pool.")

    agent = ActorCriticAgent(input_dim=int(_action_features(env, sample_actions[0]).shape[0]), device=device)
    out_dir = PROJECT_ROOT / "reports" / "rl_gpu_actor_critic"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_rows = []
    episode_rows = []
    terminal_rows = []
    for episode in range(1, args.episodes + 1):
        env.reset()
        done = False
        episode_return = 0.0
        trajectory: list[EpisodeStep] = []
        while not done:
            actions = env.available_actions()
            if not actions:
                break
            chosen, _, log_prob, entropy = agent.choose(env, actions, greedy=False)
            state_vector = torch.from_numpy(np.vstack([_action_features(env, action) for action in actions]).astype(np.float32).mean(axis=0, keepdims=True)).to(device)
            value = agent.network.value(state_vector).squeeze(0)
            _, reward, done, info = env.step(chosen.action_id)
            trajectory.append(EpisodeStep(log_prob=log_prob, value=value, reward=float(reward), entropy=entropy))
            episode_return += float(reward)
            info["episode"] = episode
            step_rows.append(info)
        loss = agent.update(trajectory)
        terminal_rows.append(env.current_observation())
        episode_rows.append(
            {
                "episode": episode,
                "episode_return": float(episode_return),
                "loss": float(loss),
                "steps": int(len(trajectory)),
            }
        )

    for seed_smiles in seeds[: max(1, int(args.evaluation_rollouts))]:
        env.reset(seed_smiles=seed_smiles)
        done = False
        while not done:
            actions = env.available_actions()
            if not actions:
                break
            chosen, _, _, _ = agent.choose(env, actions, greedy=True)
            _, _, done, _ = env.step(chosen.action_id)
        terminal_rows.append(env.current_observation())

    step_df = pd.DataFrame(step_rows)
    episode_df = pd.DataFrame(episode_rows)
    terminal_df = _postprocess(
        pd.DataFrame(terminal_rows).drop_duplicates(subset=["smiles"]).copy(),
        out_dir / "vina_poses",
        structural_top_k=max(1, int(args.structural_top_k)),
        skip_structural=bool(args.skip_structural),
    )
    terminal_df["actor_critic_priority_score"] = (
        _series(terminal_df, "verified_reward", 0.0)
        + 0.85 * _series(terminal_df, "feasibility_score", 0.0)
        + 0.40 * _series(terminal_df, "docking_rescore", 0.0)
        + 0.35 * _series(terminal_df, "interaction_support_score", 0.0)
        + 0.35 * _series(terminal_df, "cross_database_consensus_score", 0.0)
        + 0.35 * _series(terminal_df, "external_evidence_support", 0.0)
        + 0.25 * _series(terminal_df, "experimental_readiness_score", 0.0)
        + 0.28 * _series(terminal_df, "structure_evidence_support", 0.0)
        + 0.12 * _series(terminal_df, "structure_evidence_guardrail", 0.0)
        + 0.20 * _series(terminal_df, "generator_priority_score", 0.0)
        + 0.18 * _series(terminal_df, "adaptive_action_prior", 0.5)
    )
    terminal_df = terminal_df.sort_values(
        [
            "structure_evidence_state_priority" if "structure_evidence_state_priority" in terminal_df.columns else "actor_critic_priority_score",
            "structure_evidence_pareto_front_rank" if "structure_evidence_pareto_front_rank" in terminal_df.columns else "actor_critic_priority_score",
            "structure_evidence_priority" if "structure_evidence_priority" in terminal_df.columns else "actor_critic_priority_score",
            "actor_critic_priority_score",
            "predicted_pIC50",
            "QED",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)
    terminal_df["actor_critic_rank"] = terminal_df.index + 1

    step_df.to_csv(out_dir / "gpu_actor_critic_step_ledger.csv", index=False)
    episode_df.to_csv(out_dir / "gpu_actor_critic_episode_summary.csv", index=False)
    terminal_df.to_csv(out_dir / "gpu_actor_critic_top_candidates.csv", index=False)
    agent.save(out_dir / "gpu_actor_critic_state_dict.pt")
    _plot_curve(episode_df.rename(columns={"loss": "mean_loss"}), out_dir)

    summary = {
        "device": device_label,
        "episodes": int(args.episodes),
        "max_actions_per_family": int(args.max_actions_per_family),
        "max_actions_total": int(args.max_actions_total),
        "structure_guidance_budget": int(args.structure_guidance_budget),
        "best_episode_return": float(episode_df["episode_return"].max()) if not episode_df.empty else 0.0,
        "mean_episode_return": float(episode_df["episode_return"].mean()) if not episode_df.empty else 0.0,
        "mean_feasibility_score": float(terminal_df["feasibility_score"].mean()) if "feasibility_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_cross_database_consensus": float(terminal_df["cross_database_consensus_score"].mean()) if "cross_database_consensus_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_external_evidence_support": float(terminal_df["external_evidence_support"].mean()) if "external_evidence_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_adaptive_action_prior": float(terminal_df["adaptive_action_prior"].mean()) if "adaptive_action_prior" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_experimental_readiness_score": float(terminal_df["experimental_readiness_score"].mean()) if "experimental_readiness_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_structure_evidence_support": float(terminal_df["structure_evidence_support"].mean()) if "structure_evidence_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_structure_evidence_guardrail": float(terminal_df["structure_evidence_guardrail"].mean()) if "structure_evidence_guardrail" in terminal_df.columns and not terminal_df.empty else 0.0,
        "ready_rate": float((terminal_df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "structure_evidence_pass_rate": float((terminal_df["structure_evidence_status"] == "pass").mean()) if "structure_evidence_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "top_candidate": terminal_df.head(1).to_dict(orient="records"),
    }
    (out_dir / "gpu_actor_critic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved GPU actor-critic artifacts to: {out_dir}")
    if not terminal_df.empty:
        print(
            terminal_df[
                [
                    "smiles",
                    "predicted_pIC50",
                    "feasibility_score",
                    "cross_database_consensus_score",
                    "external_evidence_support",
                    "structure_evidence_support",
                    "adaptive_action_prior",
                    "actor_critic_priority_score",
                ]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()
