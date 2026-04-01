from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "presentation_visuals_polished"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0F172A",
            "font.size": 12,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
        }
    )


def _save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _card(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str, title: str, subtitle: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=28, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=14)


def build_context_cards() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6))
    ax.axis("off")
    fig.suptitle("Context Stiintific", fontsize=26, fontweight="bold", y=0.96)

    _card(ax, 0.03, 0.16, 0.28, 0.60, "#DBEAFE", "2,48 milioane", "cazuri noi de cancer pulmonar")
    _card(ax, 0.36, 0.16, 0.28, 0.60, "#DCFCE7", "10-20%", "cazuri estimate la nefumatori")
    _card(ax, 0.69, 0.16, 0.28, 0.60, "#FEF3C7", "49,3%", "prevalenta EGFR la nefumatori cu NSCLC")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "01_context_cards.png")


def build_context_bars() -> None:
    labels = ["Cancer pulmonar", "Nefumatori", "EGFR in NSCLC\nla nefumatori"]
    values = [2480675, 2480675 * 0.15, 2480675 * 0.15 * 0.493]
    colors = ["#3B82F6", "#10B981", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(12, 6.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("De ce conteaza problema?", pad=14)
    ax.set_ylabel("Numar estimat de cazuri")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{int(round(value)):,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "02_context_bars.png")


def build_data_sources_donut(multisource_summary: dict) -> None:
    labels = ["Papyrus", "ExCAPE", "BindingDB"]
    counts = [multisource_summary["papyrus_rows"], multisource_summary["excape_rows"], multisource_summary["bindingdb_rows"]]
    colors = ["#60A5FA", "#34D399", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        wedgeprops={"width": 0.38, "edgecolor": "#FFFFFF", "linewidth": 3},
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    ax.text(0, 0, "Surse\npublice", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.set_title("Ponderea surselor de date", pad=18)
    _save(fig, "03_data_sources_donut.png")


def build_data_sources_counts(multisource_summary: dict) -> None:
    labels = ["Papyrus", "ExCAPE", "BindingDB"]
    values = [multisource_summary["papyrus_rows"], multisource_summary["excape_rows"], multisource_summary["bindingdb_rows"]]
    colors = ["#60A5FA", "#34D399", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Cate inregistrari au fost folosite", pad=14)
    ax.set_ylabel("Numar de inregistrari")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "04_data_sources_counts.png")


def build_pipeline_cards() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 5.4))
    ax.axis("off")
    fig.suptitle("Cum Lucreaza Sistemul", fontsize=26, fontweight="bold", y=0.96)

    steps = [
        (0.03, "#DBEAFE", "1. Date\npublice"),
        (0.24, "#DCFCE7", "2. Model\nAI"),
        (0.45, "#FEF3C7", "3. Generare\nmolecule"),
        (0.66, "#FCE7F3", "4. Filtrare si\nverificare"),
        (0.87, "#E9D5FF", "5. Lot\nfinal"),
    ]

    y = 0.22
    w = 0.14
    h = 0.42

    for i, (x, color, text) in enumerate(steps):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=17, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 0.012, y + h / 2),
                xytext=(x + w, y + h / 2),
                arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": "#475569"},
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "05_pipeline_cards.png")


def build_results_cards(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.axis("off")
    fig.suptitle("Rezultatele Proiectului", fontsize=26, fontweight="bold", y=0.96)

    cards = [
        (0.04, 0.50, "#DBEAFE", "16.133", "molecule evaluate"),
        (0.52, 0.50, "#DCFCE7", "330", "molecule generate"),
        (0.04, 0.10, "#FEF3C7", "60", "molecule verificate"),
        (0.52, 0.10, "#FCE7F3", "18", "molecule finale"),
    ]

    for x, y, color, big, small in cards:
        _card(ax, x, y, 0.40, 0.28, color, big, small)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "06_results_cards.png")


def build_funnel_bars(metrics: dict) -> None:
    ranked = metrics["ranked_molecules"]
    generated = metrics["generated_candidate_count"] + metrics["ai_guided_candidate_count"] + metrics["iterative_candidate_count"]
    checked = metrics["vina_docked_candidates"]
    final_batch = metrics["prospective_batch_size"]

    labels = ["Evaluate", "Generate", "Verificate", "Finale"]
    values = [ranked, generated, checked, final_batch]
    colors = ["#3B82F6", "#14B8A6", "#F59E0B", "#EF4444"]

    fig, ax = plt.subplots(figsize=(12.5, 7))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Cum se restrange spatiul chimic", pad=14)
    ax.set_ylabel("Numar de molecule")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "07_funnel_bars.png")


def build_generation_paths(metrics: dict) -> None:
    labels = ["Generated", "AI-guided", "Iterative", "Lot final"]
    values = [
        metrics["generated_candidate_count"],
        metrics["ai_guided_candidate_count"],
        metrics["iterative_candidate_count"],
        metrics["prospective_batch_size"],
    ]
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171"]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Cate molecule au fost produse pe etape", pad=14)
    ax.set_ylabel("Numar de molecule")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "08_generation_paths.png")


def build_model_validation(metrics: dict) -> None:
    splits = ["Random", "Scaffold", "Temporal"]
    rmse = [metrics["model_random_rmse"], metrics["model_scaffold_rmse"], metrics["model_temporal_rmse"]]
    r2 = [
        0.7469832830172807,
        0.683374484713541,
        -0.1016866991270644,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = ["#3B82F6", "#14B8A6", "#F97316"]

    bars1 = axes[0].bar(splits, rmse, color=colors, width=0.58)
    axes[0].set_title("Eroare model")
    axes[0].set_ylabel("RMSE")
    for bar, value in zip(bars1, rmse):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

    bars2 = axes[1].bar(splits, r2, color=colors, width=0.58)
    axes[1].set_title("Generalizare")
    axes[1].set_ylabel("R²")
    axes[1].axhline(0, color="#94A3B8", lw=1.5)
    for bar, value in zip(bars2, r2):
        offset = 0.03 if value >= 0 else -0.08
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Validarea Modelului", fontsize=24, fontweight="bold", y=1.02)
    _save(fig, "09_model_validation.png")


def build_multi_agent_grouped(rediscovery: dict) -> None:
    total = rediscovery["positive_count"]
    simple = [
        round(rediscovery["naive_top10_recall"] * total),
        round(rediscovery["naive_top20_recall"] * total),
    ]
    multi = [
        round(rediscovery["protected_top10_recall"] * total),
        round(rediscovery["protected_top20_recall"] * total),
    ]

    x = np.arange(2)
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars1 = ax.bar(x - width / 2, simple, width, label="Abordare simpla", color="#F97316")
    bars2 = ax.bar(x + width / 2, multi, width, label="Multi-agent", color="#14B8A6")

    ax.set_xticks(x, ["Top 10", "Top 20"])
    ax.set_ylabel("Molecule bune gasite")
    ax.set_title("Simplu vs Multi-agent", pad=14)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{int(bar.get_height())}", ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "10_multi_agent_grouped.png")


def build_multi_agent_stacked(rediscovery: dict) -> None:
    total = rediscovery["positive_count"]
    found = [
        round(rediscovery["naive_top10_recall"] * total),
        round(rediscovery["naive_top20_recall"] * total),
        round(rediscovery["protected_top10_recall"] * total),
        round(rediscovery["protected_top20_recall"] * total),
    ]
    missed = [total - value for value in found]
    labels = ["Simplu\nTop 10", "Simplu\nTop 20", "Multi-agent\nTop 10", "Multi-agent\nTop 20"]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bars = ax.bar(labels, found, color="#2DD4BF", width=0.62, label="Gasite")
    ax.bar(labels, missed, bottom=found, color="#E5E7EB", width=0.62, label="Ratate")
    ax.set_title("Cate molecule bune sunt gasite", pad=14)
    ax.set_ylabel("Numar de molecule de referinta")
    ax.legend(frameon=False)

    for bar, value in zip(bars, found):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value}", ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "11_multi_agent_stacked.png")


def build_external_validation(source_holdout: dict) -> None:
    labels_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
    }
    filtered = [item for item in source_holdout["results"] if item["source"] != "chembl"]
    labels = [labels_map[item["source"]] for item in filtered]
    values = [item["recall_top20pct"] * 100 for item in filtered]
    colors = ["#2563EB", "#0EA5E9", "#10B981"]

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Validare pe baze externe", pad=14)
    ax.set_ylabel("Molecule puternice gasite in top 20 (%)")
    ax.set_ylim(0, 110)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "12_external_validation.png")


def build_final_batch_sources(batch_summary: dict) -> None:
    source_map = {
        "diverse": "Diverse",
        "shortlist": "Shortlist",
        "optimized_readiness": "Optimized",
        "rl": "RL",
        "generated": "Generated",
    }
    labels = [source_map[key] for key in batch_summary["batch_sources"].keys()]
    values = list(batch_summary["batch_sources"].values())
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("De unde provin cele 18 molecule finale", pad=14)
    ax.set_ylabel("Numar de molecule")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value}", ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "13_final_batch_sources.png")


def build_final_batch_status(batch_summary: dict) -> None:
    labels = ["Ready", "Supporting"]
    values = [
        batch_summary["batch_status_counts"]["ready"],
        batch_summary["batch_status_counts"]["supporting"],
    ]
    colors = ["#22C55E", "#94A3B8"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.set_title("Statusul lotului final", pad=14)
    ax.set_ylabel("Numar de molecule")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value}", ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "14_final_batch_status.png")


def build_final_batch_quality(batch_summary: dict) -> None:
    labels = ["Readiness", "Fezabilitate", "Docking", "Evidenta\nexterna"]
    values = [
        batch_summary["mean_readiness_score"],
        batch_summary["mean_feasibility_score"],
        batch_summary["mean_docking_rescore"],
        batch_summary["mean_external_evidence_support"],
    ]
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Calitatea medie a lotului final", pad=14)
    ax.set_ylabel("Scor mediu")
    ax.set_ylim(0, 1.05)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "15_final_batch_quality.png")


def build_external_validation_lollipop(source_holdout: dict) -> None:
    labels_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
    }
    filtered = [item for item in source_holdout["results"] if item["source"] != "chembl"]
    labels = [labels_map[item["source"]] for item in filtered]
    values = [item["recall_top20pct"] * 100 for item in filtered]
    colors = ["#2563EB", "#0EA5E9", "#10B981"]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))
    ax.hlines(y, 0, values, color=colors, linewidth=8, alpha=0.85)
    ax.scatter(values, y, s=280, color=colors, edgecolor="white", linewidth=2, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Molecule puternice gasite in top 20 (%)")
    ax.set_title("Validare Externa", pad=14)

    for value, ypos in zip(values, y):
        ax.text(value + 2, ypos, f"{value:.0f}%", va="center", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    _save(fig, "16_external_validation_lollipop.png")


def build_multi_agent_scoreboard(rediscovery: dict) -> None:
    total = rediscovery["positive_count"]
    simple_top10 = round(rediscovery["naive_top10_recall"] * total)
    simple_top20 = round(rediscovery["naive_top20_recall"] * total)
    multi_top10 = round(rediscovery["protected_top10_recall"] * total)
    multi_top20 = round(rediscovery["protected_top20_recall"] * total)

    fig, ax = plt.subplots(figsize=(14.5, 7.5))
    ax.axis("off")
    fig.suptitle("Test Final: Simplu vs Multi-agent", fontsize=28, fontweight="bold", y=0.96)

    _card(ax, 0.05, 0.18, 0.40, 0.62, "#FFEDD5", f"{simple_top10} / {simple_top20}", "molecule bune gasite\nTop 10 / Top 20")
    _card(ax, 0.55, 0.18, 0.40, 0.62, "#CCFBF1", f"{multi_top10} / {multi_top20}", "molecule bune gasite\nTop 10 / Top 20")

    ax.text(0.25, 0.86, "Abordare simpla", ha="center", va="center", fontsize=20, fontweight="bold", color="#C2410C")
    ax.text(0.75, 0.86, "Multi-agent", ha="center", va="center", fontsize=20, fontweight="bold", color="#0F766E")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "17_multi_agent_scoreboard.png")


def build_final_batch_sources_donut(batch_summary: dict) -> None:
    source_map = {
        "diverse": "Diverse",
        "shortlist": "Shortlist",
        "optimized_readiness": "Optimized",
        "rl": "RL",
        "generated": "Generated",
    }
    labels = [source_map[key] for key in batch_summary["batch_sources"].keys()]
    values = list(batch_summary["batch_sources"].values())
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        wedgeprops={"width": 0.38, "edgecolor": "#FFFFFF", "linewidth": 3},
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax.text(0, 0, "18\nmolecule", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.set_title("Compozitia Lotului Final", pad=18)
    _save(fig, "18_final_batch_sources_donut.png")


def build_final_batch_status_donut(batch_summary: dict) -> None:
    labels = ["Ready", "Supporting"]
    values = [
        batch_summary["batch_status_counts"]["ready"],
        batch_summary["batch_status_counts"]["supporting"],
    ]
    colors = ["#22C55E", "#94A3B8"]

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        wedgeprops={"width": 0.38, "edgecolor": "#FFFFFF", "linewidth": 3},
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    ax.text(0, 0, "Status\nfinal", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.set_title("Statusul Lotului Final", pad=18)
    _save(fig, "19_final_batch_status_donut.png")


def build_impact_funnel_cards(metrics: dict) -> None:
    ranked = metrics["ranked_molecules"]
    generated = metrics["generated_candidate_count"] + metrics["ai_guided_candidate_count"] + metrics["iterative_candidate_count"]
    checked = metrics["vina_docked_candidates"]
    final_batch = metrics["prospective_batch_size"]

    fig, ax = plt.subplots(figsize=(15.5, 6.5))
    ax.axis("off")
    fig.suptitle("De la Mii de Variante la un Lot Final", fontsize=28, fontweight="bold", y=0.96)

    cards = [
        (0.03, "#DBEAFE", f"{ranked:,}".replace(",", "."), "evaluate"),
        (0.27, "#DCFCE7", f"{generated}", "generate"),
        (0.51, "#FEF3C7", f"{checked}", "verificate"),
        (0.75, "#FCE7F3", f"{final_batch}", "finale"),
    ]

    for i, (x, color, big, small) in enumerate(cards):
        _card(ax, x, 0.25, 0.18, 0.42, color, big, small)
        if i < len(cards) - 1:
            ax.annotate(
                "",
                xy=(cards[i + 1][0] - 0.015, 0.46),
                xytext=(x + 0.18, 0.46),
                arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"},
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "20_impact_funnel_cards.png")


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(REPORTS_DIR / "technical_notebook" / "technical_notebook_metrics.json")
    rediscovery = _load_json(REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_summary.json")
    source_holdout = _load_json(REPORTS_DIR / "source_holdout_benchmark.json")
    multisource_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "egfr_multisource_summary.json")
    batch_summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")

    build_context_cards()
    build_context_bars()
    build_data_sources_donut(multisource_summary)
    build_data_sources_counts(multisource_summary)
    build_pipeline_cards()
    build_results_cards(metrics)
    build_funnel_bars(metrics)
    build_generation_paths(metrics)
    build_model_validation(metrics)
    build_multi_agent_grouped(rediscovery)
    build_multi_agent_stacked(rediscovery)
    build_external_validation(source_holdout)
    build_final_batch_sources(batch_summary)
    build_final_batch_status(batch_summary)
    build_final_batch_quality(batch_summary)
    build_external_validation_lollipop(source_holdout)
    build_multi_agent_scoreboard(rediscovery)
    build_final_batch_sources_donut(batch_summary)
    build_final_batch_status_donut(batch_summary)
    build_impact_funnel_cards(metrics)

    print(f"[OK] Saved polished presentation visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
