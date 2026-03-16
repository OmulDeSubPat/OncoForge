from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT


CHALLENGE_DIR = PROJECT_ROOT / "reports" / "reward_hacking_challenge"
RANKED_PATH = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"


def _cohort(df: pd.DataFrame, mask: pd.Series, sort_column: str, top_k: int) -> pd.DataFrame:
    cohort = df.loc[mask].copy()
    if cohort.empty:
        return cohort
    return cohort.sort_values(sort_column, ascending=False).head(top_k).copy()


def _cohort_summary(name: str, df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "cohort": name,
            "n": 0,
            "mean_naive_rank": None,
            "mean_final_rank": None,
            "mean_rank_shift": None,
            "median_rank_shift": None,
            "demoted_20plus_rate": None,
            "audit_pass_rate": None,
            "review_or_fail_rate": None,
            "veto_rate": None,
            "lead_bucket_rate": None,
            "mean_reward_hacking_risk": None,
            "mean_applicability_score": None,
            "mean_predicted_pIC50": None,
            "mean_qed": None,
        }
    rank_shift = df["rank"] - df["naive_rank"]
    return {
        "cohort": name,
        "n": int(len(df)),
        "mean_naive_rank": float(df["naive_rank"].mean()),
        "mean_final_rank": float(df["rank"].mean()),
        "mean_rank_shift": float(rank_shift.mean()),
        "median_rank_shift": float(rank_shift.median()),
        "demoted_20plus_rate": float((rank_shift >= 20).mean()),
        "audit_pass_rate": float((df["audit_status"] == "pass").mean()),
        "review_or_fail_rate": float(df["audit_status"].isin(["review", "fail"]).mean()),
        "veto_rate": float((df["veto"] == True).mean()),
        "lead_bucket_rate": float((df["selection_bucket"] == "lead").mean()),
        "mean_reward_hacking_risk": float(df["reward_hacking_risk"].mean()),
        "mean_applicability_score": float(df["applicability_score"].mean()),
        "mean_predicted_pIC50": float(df["predicted_pIC50"].mean()),
        "mean_qed": float(df["QED"].mean()),
    }


def _plot_rank_shift(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary_df.dropna(subset=["mean_rank_shift"]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(plot_df["cohort"], plot_df["mean_rank_shift"], color=["#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#6d597a"][: len(plot_df)])
    ax.axhline(0.0, color="#6c757d", linewidth=1.0)
    ax.set_ylabel("Mean demotion from naive to protected rank")
    ax.set_title("Reward-Hacking Challenge: Does Protected Ranking Push Risky Molecules Down?")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "challenge_rank_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_status_rates(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary_df.dropna(subset=["audit_pass_rate"]).copy()
    if plot_df.empty:
        return
    x = range(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - width for i in x], plot_df["audit_pass_rate"], width=width, label="Audit pass", color="#2a9d8f")
    ax.bar(x, plot_df["review_or_fail_rate"], width=width, label="Review/fail", color="#e76f51")
    ax.bar([i + width for i in x], plot_df["veto_rate"], width=width, label="Veto", color="#6d597a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["cohort"], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Reward-Hacking Challenge: Audit Outcomes by Cohort")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "challenge_status_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RANKED_PATH.exists():
        raise FileNotFoundError(f"Missing ranked dataset: {RANKED_PATH}")

    df = pd.read_csv(RANKED_PATH, low_memory=False)
    out_dir = CHALLENGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = {
        "trusted_controls": _cohort(
            df,
            (df["audit_status"] == "pass")
            & (df["veto"] == False)
            & (df["reward_hacking_risk"] <= 0.10)
            & (df["applicability_score"] >= 0.50),
            "final_score",
            50,
        ),
        "proxy_exploits": _cohort(
            df,
            (df["reward_hacking_risk"] >= 0.25)
            | (df["audit_status"] != "pass")
            | (df["veto"] == True),
            "naive_score",
            50,
        ),
        "out_of_domain_potent": _cohort(
            df,
            (df["predicted_pIC50"] >= 9.0)
            & (df["applicability_score"] < 0.25),
            "naive_score",
            50,
        ),
        "reactive_or_alerted": _cohort(
            df,
            (df["covalent_warhead_count"] > 0)
            | (df["severe_alert_count"] > 0)
            | (df["has_PAINS"] == True),
            "naive_score",
            50,
        ),
        "high_disagreement": _cohort(
            df,
            (df["agent_disagreement_score"] >= 0.45)
            & (df["naive_score"] >= df["naive_score"].quantile(0.80)),
            "naive_score",
            50,
        ),
    }

    summary_rows = [_cohort_summary(name, cohort_df) for name, cohort_df in cohorts.items()]
    summary_df = pd.DataFrame(summary_rows)

    examples = []
    for name, cohort_df in cohorts.items():
        if cohort_df.empty:
            continue
        example_df = cohort_df.sort_values(
            ["rank", "reward_hacking_risk"],
            ascending=[True, False],
        ).head(12).copy()
        example_df["cohort"] = name
        examples.append(
            example_df[
                [
                    "cohort",
                    "smiles",
                    "predicted_pIC50",
                    "QED",
                    "applicability_score",
                    "reward_hacking_risk",
                    "naive_rank",
                    "rank",
                    "audit_status",
                    "veto",
                    "selection_bucket",
                ]
            ]
        )

    examples_df = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()

    summary_path = out_dir / "reward_hacking_challenge_summary.csv"
    examples_path = out_dir / "reward_hacking_challenge_examples.csv"
    json_path = out_dir / "reward_hacking_challenge_summary.json"
    summary_df.to_csv(summary_path, index=False)
    examples_df.to_csv(examples_path, index=False)
    json_path.write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    _plot_rank_shift(summary_df, out_dir)
    _plot_status_rates(summary_df, out_dir)

    print(f"[OK] Saved reward-hacking challenge summary: {summary_path}")
    print(f"[OK] Saved reward-hacking challenge examples: {examples_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
