from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.evidence_arbiter import add_evidence_arbiter_ranking
from src.config import PROJECT_ROOT
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.experimental_readiness import add_experimental_readiness, load_market_benchmark
from src.feasibility.assessor import FeasibilityAssessor
from src.rl.environment import VerifiableMoleculeEnv
from src.rl.q_learning import TabularQLearningAgent
from src.structure.docking_rescoring import StructuralConsensusRescorer
from src.structure.interaction_analysis import PoseInteractionAnalyzer


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _select_seed_pool(ranked_path: Path, pool_size: int = 30) -> list[str]:
    df = pd.read_csv(ranked_path, low_memory=False)
    seed_df = df[
        (df["audit_status"] == "pass")
        & (df["veto"] == False)
        & (df["reward_hacking_risk"] <= 0.25)
        & (df["QED"] >= 0.35)
        & (df["predicted_pIC50"] >= 8.2)
    ].copy()
    return seed_df.sort_values("final_score", ascending=False).head(pool_size)["smiles"].tolist()


def _plot_training_curve(episode_df: pd.DataFrame, out_dir: Path) -> None:
    if episode_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episode_df["episode"], episode_df["episode_return"], color="#1d3557", linewidth=1.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Verifiable RL Training Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_reward_breakdown(step_df: pd.DataFrame, out_dir: Path) -> None:
    if step_df.empty:
        return
    cols = [
        "reward_delta_verified",
        "reward_delta_potency",
        "reward_delta_qed",
        "reward_feasibility",
        "reward_generator_priority",
        "reward_parent_similarity",
        "reward_hacking_penalty",
        "reward_veto_penalty",
    ]
    means = [float(step_df[col].mean()) for col in cols if col in step_df.columns]
    labels = [col.replace("reward_", "") for col in cols if col in step_df.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, means, color="#2a9d8f")
    ax.set_ylabel("Mean contribution")
    ax.set_title("Average Verifiable Reward Components")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_reward_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compare_to_baselines(rl_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    baseline_files = [
        ("iterative_baseline", PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_rescored.csv"),
        ("diverse_baseline", PROJECT_ROOT / "reports" / "final_diverse_candidates.csv"),
        ("ai_guided_baseline", PROJECT_ROOT / "reports" / "ai_guided_analogs.csv"),
    ]
    comparison_rows = []
    for label, path in baseline_files:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False).head(50)
        comparison_rows.append(
            {
                "cohort": label,
                "n": int(len(df)),
                "mean_pIC50": float(df["predicted_pIC50"].mean()) if "predicted_pIC50" in df.columns else 0.0,
                "mean_QED": float(df["QED"].mean()) if "QED" in df.columns else 0.0,
                "mean_reward_hacking_risk": float(df["reward_hacking_risk"].mean()) if "reward_hacking_risk" in df.columns else 0.0,
                "pass_rate": float((df["audit_status"] == "pass").mean()) if "audit_status" in df.columns else 0.0,
                "mean_external_evidence_support": float(df["external_evidence_support"].mean()) if "external_evidence_support" in df.columns else 0.0,
                "ready_rate": float((df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in df.columns else 0.0,
                "mean_evidence_arbiter_support": float(df["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in df.columns else 0.0,
            }
        )
    comparison_rows.append(
        {
            "cohort": "verifiable_rl",
            "n": int(len(rl_df)),
            "mean_pIC50": float(rl_df["predicted_pIC50"].mean()) if not rl_df.empty else 0.0,
            "mean_QED": float(rl_df["QED"].mean()) if not rl_df.empty else 0.0,
            "mean_reward_hacking_risk": float(rl_df["reward_hacking_risk"].mean()) if not rl_df.empty else 0.0,
            "pass_rate": float((rl_df["audit_status"] == "pass").mean()) if not rl_df.empty else 0.0,
            "mean_external_evidence_support": float(rl_df["external_evidence_support"].mean()) if "external_evidence_support" in rl_df.columns and not rl_df.empty else 0.0,
            "ready_rate": float((rl_df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in rl_df.columns and not rl_df.empty else 0.0,
            "mean_evidence_arbiter_support": float(rl_df["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in rl_df.columns and not rl_df.empty else 0.0,
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "rl_vs_baselines.csv", index=False)
    return comparison


def _plot_external_evidence_priority(rl_df: pd.DataFrame, out_dir: Path) -> None:
    if rl_df.empty or "external_evidence_support" not in rl_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        rl_df["external_evidence_support"],
        rl_df["rl_priority_score"],
        c=rl_df.get("cross_database_consensus_score", pd.Series(0.0, index=rl_df.index)),
        cmap="viridis",
        alpha=0.75,
        s=28,
    )
    ax.axvline(0.45, linestyle="--", linewidth=1.0, color="#6c757d")
    ax.set_xlabel("External evidence support")
    ax.set_ylabel("RL priority score")
    ax.set_title("RL Candidates: Public Evidence vs Final RL Priority")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Cross-database consensus")
    fig.tight_layout()
    fig.savefig(out_dir / "rl_external_evidence_vs_priority.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _evaluation_rollout(
    env: VerifiableMoleculeEnv,
    agent: TabularQLearningAgent,
    seed_smiles: str,
) -> dict:
    state_key = env.reset(seed_smiles=seed_smiles)
    done = False
    while not done:
        actions = env.available_actions()
        action_names = [action.action_id for action in actions]
        if not action_names:
            break
        action_priors = {
            action.action_id: float(action.reward_profile["reward_total"]) + 0.10 * float(action.candidate_profile.get("generator_priority_score", 0.0))
            for action in actions
        }
        action_name = agent.select_action(state_key, action_names, greedy=True, action_priors=action_priors)
        state_key, _, done, _ = env.step(action_name)
    return env.current_observation()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a verifiable RL optimizer over medicinal-chemistry actions.")
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed-pool-size", type=int, default=30)
    parser.add_argument("--evaluation-rollouts", type=int, default=12)
    parser.add_argument("--max-actions-per-family", type=int, default=3)
    parser.add_argument("--max-actions-total", type=int, default=24)
    args = parser.parse_args(argv)

    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing ranked dataset: {ranked_path}")

    seeds = _select_seed_pool(ranked_path, pool_size=args.seed_pool_size)
    env = VerifiableMoleculeEnv(
        seeds,
        assessor=FeasibilityAssessor(),
        max_steps=args.max_steps,
        max_actions_per_family=args.max_actions_per_family,
        max_actions_total=args.max_actions_total,
    )
    agent = TabularQLearningAgent()

    rl_dir = PROJECT_ROOT / "reports" / "rl_verifiable"
    rl_dir.mkdir(parents=True, exist_ok=True)

    step_rows = []
    episode_rows = []
    terminal_rows = []

    for episode in range(1, args.episodes + 1):
        state_key = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            actions = env.available_actions()
            action_names = [action.action_id for action in actions]
            if not action_names:
                break
            action_priors = {
                action.action_id: float(action.reward_profile["reward_total"]) + 0.10 * float(action.candidate_profile.get("generator_priority_score", 0.0))
                for action in actions
            }
            action_name = agent.select_action(state_key, action_names, action_priors=action_priors)
            next_state_key, reward, done, info = env.step(action_name)
            next_actions = env.available_actions()
            next_action_names = [action.action_id for action in next_actions]
            agent.update(state_key, action_name, reward, next_state_key, next_action_names, done)
            agent.replay()
            episode_return += reward
            info["episode"] = episode
            step_rows.append(info)
            state_key = next_state_key

        observation = env.current_observation()
        episode_rows.append(
            {
                "episode": episode,
                "episode_return": episode_return,
                "terminal_smiles": observation.get("smiles"),
                "terminal_pIC50": observation.get("predicted_pIC50"),
                "terminal_QED": observation.get("QED"),
                "terminal_verified_reward": observation.get("verified_reward"),
                "terminal_audit_status": observation.get("audit_status"),
            }
        )
        terminal_rows.append(observation)
        agent.decay_epsilon()

    for seed_smiles in seeds[: max(1, int(args.evaluation_rollouts))]:
        terminal_rows.append(_evaluation_rollout(env, agent, seed_smiles))

    step_df = pd.DataFrame(step_rows)
    episode_df = pd.DataFrame(episode_rows)
    terminal_df = pd.DataFrame(terminal_rows).drop_duplicates(subset=["smiles"]).copy()

    rescorer = StructuralConsensusRescorer(
        backend="auto",
        pose_dir=PROJECT_ROOT / "reports" / "vina_poses" / "rl_candidates",
        vina_cpu=1,
        vina_exhaustiveness=6,
        vina_num_modes=5,
    )
    analyzer = PoseInteractionAnalyzer()
    if rescorer.is_available() and not terminal_df.empty:
        terminal_df = terminal_df.sort_values("verified_reward", ascending=False).head(20).copy()
        structural_rows = []
        for idx, (_, row) in enumerate(terminal_df.iterrows(), start=1):
            rescored = rescorer.score_smiles(str(row["smiles"]), ligand_name=f"rl_{idx:03d}")
            out_row = row.to_dict()
            out_row.update(rescored)
            pose_path = rescored.get("docking_pose_path")
            if isinstance(pose_path, str) and pose_path:
                out_row.update(analyzer.analyze_pose(pose_path, smiles=str(row["smiles"])))
            structural_rows.append(out_row)
        terminal_df = pd.DataFrame(structural_rows)

    assessor = FeasibilityAssessor()
    feasibility_rows = []
    for _, row in terminal_df.iterrows():
        feasibility = assessor.assess(
            str(row["smiles"]),
            docking_rescore=float(row["docking_rescore"]) if "docking_rescore" in row and pd.notna(row["docking_rescore"]) else None,
            interaction_support_score=float(row["interaction_support_score"]) if "interaction_support_score" in row and pd.notna(row["interaction_support_score"]) else None,
            interaction_key_residue_count=int(row["interaction_key_residue_count"]) if "interaction_key_residue_count" in row and pd.notna(row["interaction_key_residue_count"]) else None,
        )
        out_row = row.to_dict()
        out_row.update(feasibility)
        out_row["rl_priority_score"] = (
            float(out_row.get("verified_reward", 0.0))
            + 0.80 * float(out_row["feasibility_score"])
            + 0.50 * float(out_row.get("docking_rescore", 0.0))
            + 0.50 * float(out_row.get("interaction_support_score", 0.0))
        )
        feasibility_rows.append(out_row)
    terminal_df = pd.DataFrame(feasibility_rows)
    if not terminal_df.empty:
        validator = CrossDatabaseValidator()
        terminal_df = validator.validate_frame(terminal_df)
        terminal_df = add_experimental_readiness(
            terminal_df,
            market_df=load_market_benchmark(),
            sort_output=False,
        )
        terminal_df = add_evidence_arbiter_ranking(terminal_df)
        terminal_df["rl_priority_score"] = (
            _series(terminal_df, "verified_reward", 0.0)
            + 0.80 * _series(terminal_df, "feasibility_score", 0.0)
            + 0.45 * _series(terminal_df, "docking_rescore", 0.0)
            + 0.45 * _series(terminal_df, "interaction_support_score", 0.0)
            + 0.30 * _series(terminal_df, "cross_database_consensus_score", 0.0)
            + 0.30 * _series(terminal_df, "external_evidence_support", 0.0)
            + 0.25 * _series(terminal_df, "experimental_readiness_score", 0.0)
            + 0.25 * _series(terminal_df, "evidence_arbiter_support", 0.0)
        )
        terminal_df["rl_audit_priority"] = terminal_df.get("audit_status", pd.Series("review", index=terminal_df.index)).map(
            {"pass": 0, "review": 1, "fail": 2}
        ).fillna(1).astype(int)
        terminal_df["rl_external_priority"] = terminal_df.get(
            "external_evidence_status",
            pd.Series("review", index=terminal_df.index),
        ).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
        terminal_df["rl_readiness_priority"] = terminal_df.get(
            "experimental_readiness_status",
            pd.Series("supporting", index=terminal_df.index),
        ).map({"ready": 0, "supporting": 1, "hold": 2}).fillna(1).astype(int)
        terminal_df["rl_arbiter_priority"] = terminal_df.get(
            "evidence_arbiter_status",
            pd.Series("review", index=terminal_df.index),
        ).map({"pass": 0, "review": 1, "fail": 2}).fillna(1).astype(int)
    terminal_df = terminal_df.sort_values(
        [
            "rl_arbiter_priority" if "rl_arbiter_priority" in terminal_df.columns else "rl_priority_score",
            "rl_audit_priority" if "rl_audit_priority" in terminal_df.columns else "audit_status",
            "rl_external_priority" if "rl_external_priority" in terminal_df.columns else "rl_priority_score",
            "rl_readiness_priority" if "rl_readiness_priority" in terminal_df.columns else "rl_priority_score",
            "rl_priority_score",
            "docking_rescore",
            "predicted_pIC50",
            "QED",
        ],
        ascending=[True, True, True, True, False, False, False, False],
    ).reset_index(drop=True)
    terminal_df["rl_rank"] = terminal_df.index + 1

    step_df.to_csv(rl_dir / "rl_step_ledger.csv", index=False)
    episode_df.to_csv(rl_dir / "rl_episode_summary.csv", index=False)
    terminal_df.to_csv(rl_dir / "rl_top_candidates.csv", index=False)
    terminal_df.to_csv(rl_dir / "rl_top_candidates_crossdb.csv", index=False)
    agent.save(rl_dir / "rl_q_table.json")

    comparison = _compare_to_baselines(terminal_df.head(50), rl_dir)
    _plot_training_curve(episode_df, rl_dir)
    _plot_reward_breakdown(step_df, rl_dir)
    _plot_external_evidence_priority(terminal_df.head(50), rl_dir)

    summary = {
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "max_actions_per_family": int(args.max_actions_per_family),
        "max_actions_total": int(args.max_actions_total),
        "seed_pool_size": int(len(seeds)),
        "best_episode_return": float(episode_df["episode_return"].max()) if not episode_df.empty else 0.0,
        "mean_episode_return": float(episode_df["episode_return"].mean()) if not episode_df.empty else 0.0,
        "mean_generator_priority_score": float(terminal_df["generator_priority_score"].mean()) if "generator_priority_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_parent_similarity": float(terminal_df["parent_similarity"].mean()) if "parent_similarity" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_cross_database_consensus": float(terminal_df["cross_database_consensus_score"].mean()) if "cross_database_consensus_score" in terminal_df.columns and not terminal_df.empty else 0.0,
        "external_evidence_pass_rate": float((terminal_df["external_evidence_status"] == "pass").mean()) if "external_evidence_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_external_evidence_support": float(terminal_df["external_evidence_support"].mean()) if "external_evidence_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "readiness_ready_rate": float((terminal_df["experimental_readiness_status"] == "ready").mean()) if "experimental_readiness_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "arbiter_pass_rate": float((terminal_df["evidence_arbiter_status"] == "pass").mean()) if "evidence_arbiter_status" in terminal_df.columns and not terminal_df.empty else 0.0,
        "mean_evidence_arbiter_support": float(terminal_df["evidence_arbiter_support"].mean()) if "evidence_arbiter_support" in terminal_df.columns and not terminal_df.empty else 0.0,
        "top_candidate": terminal_df.head(1).to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
    }
    (rl_dir / "rl_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved RL artifacts to: {rl_dir}")
    if not terminal_df.empty:
        print(
            terminal_df[
                [
                    "smiles",
                    "predicted_pIC50",
                    "QED",
                    "feasibility_score",
                    "docking_rescore",
                    "interaction_support_score",
                    "cross_database_consensus_score",
                    "external_evidence_support",
                    "experimental_readiness_score",
                    "evidence_arbiter_support",
                    "audit_status",
                    "reward_hacking_risk",
                    "rl_priority_score",
                ]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()
