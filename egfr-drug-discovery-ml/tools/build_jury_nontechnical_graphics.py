from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_DIR = PROJECT_ROOT / "grafice_juriu_non_tehnic_30-3-2026" / "grafice_create_special"


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#111827",
            "font.size": 13,
            "axes.titlesize": 23,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "legend.fontsize": 12,
        }
    )


def _save(fig: plt.Figure, filename: str) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(TARGET_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_selection_story() -> None:
    ranked_count = len(pd.read_csv(REPORTS_DIR / "ranked_egfr_dataset.csv", low_memory=False))
    summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    pool_count = int(summary["candidate_pool_size"])
    final_count = int(summary["selected_batch_size"])

    labels = ["Evaluate de model", "Intrate in pool", "Lot final"]
    values = [ranked_count, pool_count, final_count]
    colors = ["#4F8EDC", "#34C38F", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("Cum reducem mii de molecule la un lot final")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            _format_int(int(value)),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.92,
        f"Din {_format_int(ranked_count)} molecule evaluate, doar {_format_int(final_count)} ajung in lotul final.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )
    _save(fig, "11_de_la_mii_la_lotul_final.png")


def build_multi_agent_advantage() -> None:
    recall_df = pd.read_csv(REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_recall_at_k.csv", low_memory=False)
    recall_df = recall_df[recall_df["k"].isin([10, 20])].copy()
    recall_df["Protejat"] = recall_df["protected_recall"] * 100.0
    recall_df["Naiv"] = recall_df["naive_recall"] * 100.0

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = range(len(recall_df))
    width = 0.34
    protected = ax.bar([i - width / 2 for i in x], recall_df["Protejat"], width, color="#12B981", label="Selectie multi-agent")
    naive = ax.bar([i + width / 2 for i in x], recall_df["Naiv"], width, color="#F97316", label="Selectie simpla")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Top {int(k)}" for k in recall_df["k"]])
    ax.set_ylabel("Molecule bune gasite (%)")
    ax.set_ylim(0, max(recall_df["Protejat"].max(), 5) * 1.35)
    ax.set_title("De ce ne ajuta abordarea multi-agent")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [protected, naive]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.3,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    _save(fig, "12_de_ce_multi_agent_este_util.png")


def build_external_validation() -> None:
    source_df = pd.read_csv(REPORTS_DIR / "source_holdout_benchmark.csv", low_memory=False)
    keep = ["excape_chembl20", "papyrus", "bindingdb_articles"]
    source_df = source_df[source_df["source"].isin(keep)].copy()
    label_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
    }
    source_df["label"] = source_df["source"].map(label_map)
    source_df["recall_pct"] = source_df["recall_top20pct"] * 100.0

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    bars = ax.bar(source_df["label"], source_df["recall_pct"], color=["#2563EB", "#0EA5E9", "#10B981"], width=0.60)
    ax.set_title("Modelul ramane util si pe date externe")
    ax.set_ylabel("Molecule puternice gasite in primele 20% rezultate")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, source_df["recall_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.93,
        "Acesta este un test pe surse independente, nu doar pe datele folosite la antrenare.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    _save(fig, "13_validare_pe_date_externe.png")


def build_final_batch_status() -> None:
    summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    ready = int(summary["batch_status_counts"]["ready"])
    supporting = int(summary["batch_status_counts"]["supporting"])
    total = ready + supporting

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    colors = ["#22C55E", "#94A3B8"]
    wedges, _ = ax.pie(
        [ready, supporting],
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 4},
    )
    ax.text(0, 0.10, f"{ready}/{total}", ha="center", va="center", fontsize=28, fontweight="bold")
    ax.text(0, -0.12, "mai pregatite", ha="center", va="center", fontsize=15)
    ax.set_title("Cat de pregatit este lotul final", pad=16)
    ax.legend(
        wedges,
        [f"Ready: {ready}", f"Supporting: {supporting}"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
    )
    _save(fig, "14_cat_de_pregatit_este_lotul_final.png")


def build_final_batch_sources() -> None:
    summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    source_map = {
        "diverse": "Selectie diversa",
        "shortlist": "Shortlist",
        "optimized_readiness": "Optimizare",
        "rl": "RL",
        "generated": "Generare",
    }
    order = ["diverse", "shortlist", "optimized_readiness", "rl", "generated"]
    labels = [source_map[key] for key in order]
    values = [int(summary["batch_sources"].get(key, 0)) for key in order]

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    bars = ax.bar(labels, values, color=["#4F8EDC", "#34C38F", "#A78BFA", "#FB7185", "#FBBF24"], width=0.60)
    ax.set_title("Din ce strategii vin moleculele finale")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.22,
            str(value),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    _save(fig, "15_din_ce_strategii_vin_finalistele.png")


def build_final_batch_profile() -> None:
    summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    labels = [
        "Fezabilitate",
        "Pregatire experimentala",
        "Sprijin structural",
        "Dovezi externe",
    ]
    values = [
        float(summary["mean_feasibility_score"]) * 100.0,
        float(summary["mean_readiness_score"]) * 100.0,
        float(summary["mean_structure_evidence_support"]) * 100.0,
        float(summary["mean_external_evidence_support"]) * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    bars = ax.barh(labels, values, color=["#22C55E", "#3B82F6", "#8B5CF6", "#14B8A6"], height=0.58)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Scor mediu (%)")
    ax.set_title("Profilul mediu al lotului final")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.text(
        0.98,
        0.07,
        "Mai mare = mai bine",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        color="#475569",
    )
    _save(fig, "16_profilul_mediu_al_lotului_final.png")


def main() -> None:
    _style()
    build_selection_story()
    build_multi_agent_advantage()
    build_external_validation()
    build_final_batch_status()
    build_final_batch_sources()
    build_final_batch_profile()
    print(f"[OK] Graficele pentru juriu au fost salvate in: {TARGET_DIR}")


if __name__ == "__main__":
    main()
