from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_directml
except ImportError:  # pragma: no cover - optional backend
    torch_directml = None

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.agents.structure_evidence_arbiter import add_structure_evidence_arbiter
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.assessor import FeasibilityAssessor
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark
from src.rl.environment import GroundedAction, VerifiableMoleculeEnv
from src.structure.docking_rescoring import StructuralConsensusRescorer
from src.structure.interaction_analysis import PoseInteractionAnalyzer


ACTION_CATEGORY_ORDER = [
    "atom_swap",
    "append_group",
    "hetero_edit",
    "mmp",
    "snar",
    "acylation",
    "alkylation",
    "functional_group_swap",
    "reaction_transform",
]
RULE_SOURCE_ORDER = [
    "atom_edit",
    "append_group",
    "hetero_edit",
    "matched_molecular_pair",
    "reaction_transform",
    "medchem_edit",
]


def _resolve_device(prefer_gpu: bool = True) -> tuple[torch.device, str]:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if prefer_gpu and torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _select_seed_pool(pool_size: int = 32) -> list[str]:
    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    df = pd.read_csv(ranked_path, low_memory=False)
    seed_df = df[
        (df["audit_status"] == "pass")
        & (df["veto"] == False)
        & (df["reward_hacking_risk"] <= 0.25)
        & (df["QED"] >= 0.35)
        & (df["predicted_pIC50"] >= 8.1)
    ].copy()
    return seed_df.sort_values("final_score", ascending=False).head(pool_size)["smiles"].tolist()


def _one_hot(value: str, categories: list[str]) -> list[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def _action_features(env: VerifiableMoleculeEnv, action: GroundedAction) -> np.ndarray:
    current = env.current_profile or {}
    current_feas = env.current_feasibility or {}
    candidate = action.candidate_profile
    feasibility = action.feasibility_profile
    reward = action.reward_profile
    vector = [
        float(current.get("predicted_pIC50", 0.0)),
        float(current.get("QED", 0.0)),
        float(current.get("verified_reward", 0.0)),
        float(current.get("reward_hacking_risk", 0.0)),
        float(current_feas.get("feasibility_score", 0.0)),
        float(candidate.get("predicted_pIC50", 0.0)),
        float(candidate.get("QED", 0.0)),
        float(candidate.get("final_score", 0.0)),
        float(candidate.get("verified_reward", 0.0)),
        float(candidate.get("reward_hacking_risk", 0.0)),
        float(candidate.get("applicability_score", 0.0)),
        float(candidate.get("novelty_score", 0.0)),
        float(candidate.get("uncertainty", 0.0)),
        float(candidate.get("generator_priority_score", 0.0)),
        float(candidate.get("adaptive_action_prior", 0.5)),
        float(candidate.get("parent_similarity", 0.0)),
        float(candidate.get("property_support_score", 0.0)),
        float(candidate.get("structural_guidance_score", 0.0)),
        float(candidate.get("potency_support", 0.0)),
        float(candidate.get("chemistry_support", 0.0)),
        float(candidate.get("safety_support", 0.0)),
        float(candidate.get("domain_support", 0.0)),
        float(feasibility.get("feasibility_score", 0.0)),
        float(bool(feasibility.get("feasibility_hard_gate_pass", False))),
        float(feasibility.get("synthetic_ease_score", 0.0)),
        float(feasibility.get("route_synthetic_support_score", 0.0)),
        float(feasibility.get("medchem_realism_score", 0.0)),
        float(feasibility.get("transformation_confidence_score", 0.0)),
        float(reward.get("reward_total", 0.0)),
        float(reward.get("reward_delta_verified", 0.0)),
        float(reward.get("reward_delta_potency", 0.0)),
        float(reward.get("reward_delta_qed", 0.0)),
        float(reward.get("reward_structural_support", 0.0)),
        float(reward.get("reward_interaction_support", 0.0)),
        float(reward.get("reward_generator_priority", 0.0)),
        float(reward.get("reward_adaptive_action_prior", 0.0)),
        float(reward.get("reward_parent_similarity", 0.0)),
    ]
    vector.extend(_one_hot(str(action.action_category), ACTION_CATEGORY_ORDER))
    vector.extend(_one_hot(str(candidate.get("action_rule_source", "")), RULE_SOURCE_ORDER))
    return np.asarray(vector, dtype=np.float32)


@dataclass(frozen=True)
class Experience:
    features: np.ndarray
    reward: float
    next_features: tuple[np.ndarray, ...]
    done: bool


class QNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.08),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralQAgent:
    def __init__(self, input_dim: int, device: torch.device):
        self.device = device
        self.network = QNet(input_dim).to(device)
        self.target = QNet(input_dim).to(device)
        self.target.load_state_dict(self.network.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=8e-4, weight_decay=1e-4)
        self.memory: deque[Experience] = deque(maxlen=8192)
        self.gamma = 0.92
        self.epsilon = 0.30
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.03
        self.batch_size = 96
        self.random = random.Random(42)
        self.train_steps = 0

    def choose(self, env: VerifiableMoleculeEnv, actions: list[GroundedAction], greedy: bool = False) -> tuple[GroundedAction, np.ndarray]:
        feature_matrix = np.vstack([_action_features(env, action) for action in actions]).astype(np.float32)
        if (not greedy) and (self.random.random() < self.epsilon):
            idx = self.random.randrange(len(actions))
            return actions[idx], feature_matrix[idx]
        with torch.no_grad():
            q = self.network(torch.from_numpy(feature_matrix).to(self.device)).detach().cpu().numpy()
        priors = np.asarray(
            [
                float(action.reward_profile.get("reward_total", 0.0))
                + 0.10 * float(action.candidate_profile.get("generator_priority_score", 0.0))
                + 0.06 * float(action.candidate_profile.get("adaptive_action_prior", 0.5))
                for action in actions
            ],
            dtype=float,
        )
        idx = int(np.argmax(q + 0.08 * priors))
        return actions[idx], feature_matrix[idx]

    def store(self, exp: Experience) -> None:
        self.memory.append(exp)

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        sample = self.random.sample(list(self.memory), k=self.batch_size)
        features = torch.from_numpy(np.vstack([exp.features for exp in sample]).astype(np.float32)).to(self.device)
        rewards = torch.tensor([exp.reward for exp in sample], dtype=torch.float32, device=self.device)
        pred_q = self.network(features)
        targets = []
        with torch.no_grad():
            for exp in sample:
                if exp.done or not exp.next_features:
                    targets.append(exp.reward)
                    continue
                next_batch = torch.from_numpy(np.vstack(exp.next_features).astype(np.float32)).to(self.device)
                targets.append(exp.reward + self.gamma * float(self.target(next_batch).max().item()))
        target_q = torch.tensor(targets, dtype=torch.float32, device=self.device)
        loss = F.smooth_l1_loss(pred_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % 64 == 0:
            self.target.load_state_dict(self.network.state_dict())
        return float(loss.detach().cpu().item())

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path) -> None:
        torch.save(self.network.state_dict(), path)


def _postprocess(terminal_df: pd.DataFrame, pose_dir, structural_top_k: int, skip_structural: bool) -> pd.DataFrame:
    if terminal_df.empty:
        return terminal_df
    rescorer = StructuralConsensusRescorer(backend="auto", pose_dir=pose_dir, vina_cpu=1, vina_exhaustiveness=6, vina_num_modes=5)
    analyzer = PoseInteractionAnalyzer()
    if (not skip_structural) and rescorer.is_available():
        rows = []
        for idx, (_, row) in enumerate(terminal_df.sort_values("verified_reward", ascending=False).head(max(1, int(structural_top_k))).iterrows(), start=1):
            rescored = rescorer.score_smiles(str(row["smiles"]), ligand_name=f"gpu_rl_{idx:03d}")
            out_row = row.to_dict()
            out_row.update(rescored)
            pose_path = rescored.get("docking_pose_path")
            if isinstance(pose_path, str) and pose_path:
                out_row.update(analyzer.analyze_pose(pose_path, smiles=str(row["smiles"])))
            rows.append(out_row)
        terminal_df = pd.DataFrame(rows)
    assessor = FeasibilityAssessor()
    rows = []
    for _, row in terminal_df.iterrows():
        feasibility = assessor.assess(
            str(row["smiles"]),
            action_name=str(row["action_name"]) if "action_name" in row and pd.notna(row["action_name"]) else None,
            action_rule_source=str(row["action_rule_source"]) if "action_rule_source" in row and pd.notna(row["action_rule_source"]) else None,
            synthetic_route=str(row["synthetic_route"]) if "synthetic_route" in row and pd.notna(row["synthetic_route"]) else None,
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
        rows.append(out_row)
    terminal_df = pd.DataFrame(rows)
    terminal_df = CrossDatabaseValidator().validate_frame(terminal_df)
    terminal_df = add_experimental_readiness(terminal_df, market_df=load_market_benchmark(), sort_output=False)
    terminal_df = add_evidence_arbiter_ranking(terminal_df)
    terminal_df = add_structure_evidence_arbiter(terminal_df)
    terminal_df["gpu_rl_priority_score"] = (
        _series(terminal_df, "verified_reward", 0.0)
        + 0.80 * _series(terminal_df, "feasibility_score", 0.0)
        + 0.40 * _series(terminal_df, "docking_rescore", 0.0)
        + 0.35 * _series(terminal_df, "interaction_support_score", 0.0)
        + 0.30 * _series(terminal_df, "cross_database_consensus_score", 0.0)
        + 0.30 * _series(terminal_df, "external_evidence_support", 0.0)
        + 0.28 * _series(terminal_df, "experimental_readiness_score", 0.0)
        + 0.30 * _series(terminal_df, "evidence_arbiter_support", 0.0)
        + 0.32 * _series(terminal_df, "structure_evidence_support", 0.0)
        + 0.12 * _series(terminal_df, "structure_evidence_guardrail", 0.0)
        + 0.16 * _series(terminal_df, "adaptive_action_prior", 0.5)
    )
    terminal_df["gpu_rl_arbiter_priority"] = terminal_df.get("evidence_arbiter_status", pd.Series("review", index=terminal_df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    terminal_df["gpu_rl_structure_priority"] = terminal_df.get("structure_evidence_status", pd.Series("review", index=terminal_df.index)).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    terminal_df = terminal_df.sort_values(
        ["gpu_rl_structure_priority", "gpu_rl_arbiter_priority", "gpu_rl_priority_score", "predicted_pIC50"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    terminal_df["gpu_rl_rank"] = terminal_df.index + 1
    return terminal_df


def _plot_curve(episode_df: pd.DataFrame, out_dir) -> None:
    if episode_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episode_df["episode"], episode_df["episode_return"], color="#1d3557", linewidth=1.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("GPU DQN Verifiable RL")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_rl_training_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a GPU DQN over verifiable medicinal-chemistry actions.")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed-pool-size", type=int, default=32)
    parser.add_argument("--evaluation-rollouts", type=int, default=14)
    parser.add_argument("--structural-top-k", type=int, default=12)
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-actions-per-family", type=int, default=3)
    parser.add_argument("--max-actions-total", type=int, default=24)
    parser.add_argument("--structure-guidance-budget", type=int, default=36)
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
    agent = NeuralQAgent(input_dim=int(_action_features(env, sample_actions[0]).shape[0]), device=device)
    rl_dir = PROJECT_ROOT / "reports" / "rl_gpu_dqn"
    rl_dir.mkdir(parents=True, exist_ok=True)

    step_rows = []
    episode_rows = []
    terminal_rows = []
    for episode in range(1, args.episodes + 1):
        env.reset()
        done = False
        episode_return = 0.0
        losses = []
        while not done:
            actions = env.available_actions()
            if not actions:
                break
            chosen, features = agent.choose(env, actions, greedy=False)
            _, reward, done, info = env.step(chosen.action_id)
            next_actions = env.available_actions()
            next_features = tuple(_action_features(env, action) for action in next_actions) if next_actions else tuple()
            agent.store(Experience(features=features, reward=float(reward), next_features=next_features, done=bool(done)))
            losses.append(agent.train_step())
            episode_return += float(reward)
            info["episode"] = episode
            step_rows.append(info)
        terminal_rows.append(env.current_observation())
        episode_rows.append({"episode": episode, "episode_return": float(episode_return), "mean_loss": float(np.mean([loss for loss in losses if loss > 0.0])) if any(loss > 0.0 for loss in losses) else 0.0, "epsilon": float(agent.epsilon)})
        agent.decay()

    for seed_smiles in seeds[: max(1, int(args.evaluation_rollouts))]:
        env.reset(seed_smiles=seed_smiles)
        done = False
        while not done:
            actions = env.available_actions()
            if not actions:
                break
            chosen, _ = agent.choose(env, actions, greedy=True)
            _, _, done, _ = env.step(chosen.action_id)
        terminal_rows.append(env.current_observation())

    step_df = pd.DataFrame(step_rows)
    episode_df = pd.DataFrame(episode_rows)
    terminal_df = _postprocess(
        pd.DataFrame(terminal_rows).drop_duplicates(subset=["smiles"]).copy(),
        rl_dir / "vina_poses",
        structural_top_k=max(1, int(args.structural_top_k)),
        skip_structural=bool(args.skip_structural),
    )
    step_df.to_csv(rl_dir / "gpu_rl_step_ledger.csv", index=False)
    episode_df.to_csv(rl_dir / "gpu_rl_episode_summary.csv", index=False)
    terminal_df.to_csv(rl_dir / "gpu_rl_top_candidates.csv", index=False)
    agent.save(rl_dir / "gpu_dqn_state_dict.pt")
    _plot_curve(episode_df, rl_dir)
    summary = {
        "device": device_label,
        "episodes": int(args.episodes),
        "max_actions_per_family": int(args.max_actions_per_family),
        "max_actions_total": int(args.max_actions_total),
        "structure_guidance_budget": int(args.structure_guidance_budget),
        "best_episode_return": float(episode_df["episode_return"].max()) if not episode_df.empty else 0.0,
        "mean_episode_return": float(episode_df["episode_return"].mean()) if not episode_df.empty else 0.0,
        "mean_generator_priority_score": float(terminal_df["generator_priority_score"].mean()) if "generator_priority_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_parent_similarity": float(terminal_df["parent_similarity"].mean()) if "parent_similarity" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_adaptive_action_prior": float(terminal_df["adaptive_action_prior"].mean()) if "adaptive_action_prior" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_cross_database_consensus": float(terminal_df["cross_database_consensus_score"].mean()) if "cross_database_consensus_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_external_evidence_support": float(terminal_df["external_evidence_support"].mean()) if "external_evidence_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_experimental_readiness_score": float(terminal_df["experimental_readiness_score"].mean()) if "experimental_readiness_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "ready_rate": float((terminal_df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_evidence_arbiter_support": float(terminal_df["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_structure_evidence_support": float(terminal_df["structure_evidence_support"].mean()) if "structure_evidence_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_structure_evidence_guardrail": float(terminal_df["structure_evidence_guardrail"].mean()) if "structure_evidence_guardrail" in terminal_df.columns and not terminal_df.empty else 0.0,
        "arbiter_pass_rate": float((terminal_df["evidence_arbiter_status"] == "pass").mean()) if "evidence_arbiter_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "structure_evidence_pass_rate": float((terminal_df["structure_evidence_status"] == "pass").mean()) if "structure_evidence_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "top_candidate": terminal_df.head(1).to_dict(orient="records"),
    }
    (rl_dir / "gpu_rl_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved GPU RL artifacts to: {rl_dir}")
    if not terminal_df.empty:
        print(
            terminal_df[
                [
                    "smiles",
                    "predicted_pIC50",
                    "feasibility_score",
                    "cross_database_consensus_score",
                    "external_evidence_support",
                    "evidence_arbiter_support",
                    "structure_evidence_support",
                    "adaptive_action_prior",
                    "gpu_rl_priority_score",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
