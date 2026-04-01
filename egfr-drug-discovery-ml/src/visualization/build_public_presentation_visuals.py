from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "presentation_visuals_public"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 13,
        }
    )


def build_context_cards() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle("Context Stiintific", fontsize=24, fontweight="bold", y=0.98)

    cards = [
        ("2.480.675", "cazuri noi de cancer pulmonar", "#DBEAFE"),
        ("717.211", "cazuri de adenocarcinom la barbati", "#DCFCE7"),
        ("541.971", "cazuri de adenocarcinom la femei", "#FEF3C7"),
    ]

    for ax, (big, subtitle, color) in zip(axes, cards):
        ax.set_facecolor(color)
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=28, fontweight="bold")
        ax.text(0.5, 0.35, subtitle, ha="center", va="center", fontsize=14, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.5,
        0.03,
        "Sursa: IARC GLOBOCAN 2022.",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUTPUT_DIR / "00_context_stiintific.png", dpi=220)
    plt.close(fig)


def build_public_sources_chart(summary: dict) -> None:
    labels = ["Papyrus", "ExCAPE", "BindingDB"]
    raw_values = [summary["papyrus_rows"], summary["excape_rows"], summary["bindingdb_rows"]]
    total = sum(raw_values)
    values = [(value / total) * 100 for value in raw_values]
    colors = ["#60A5FA", "#34D399", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Date publice folosite in proiect", fontweight="bold")
    ax.set_ylabel("Procent din sursele afisate")
    ax.set_ylim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_surse_date_publice.png", dpi=220)
    plt.close(fig)


def build_simple_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Cum lucreaza sistemul", fontsize=24, fontweight="bold", y=0.96)

    boxes = [
        (0.02, "#DBEAFE", "1. Colectam\ndate publice"),
        (0.215, "#DCFCE7", "2. Antrenam\nmodelul AI"),
        (0.41, "#FEF3C7", "3. Generam si\nfiltram molecule"),
        (0.605, "#FCE7F3", "4. Verificam\ncandidatii"),
        (0.80, "#E9D5FF", "5. Alegem\nlotul\nfinal"),
    ]

    y = 0.5
    width = 0.14
    height = 0.34

    for i, (x, color, text) in enumerate(boxes):
        rect = plt.Rectangle((x, y - height / 2), width, height, facecolor=color, edgecolor="#334155", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=14, fontweight="bold")
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[i + 1][0] - 0.008, y),
                xytext=(x + width, y),
                arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#475569"},
            )

    fig.tight_layout(rect=(0.01, 0, 0.99, 0.92))
    fig.savefig(OUTPUT_DIR / "02_cum_lucreaza_sistemul.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def build_results_summary(metrics: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Rezultatele Proiectului, pe Scurt", fontsize=24, fontweight="bold", y=0.98)

    cards = [
        ("16.133", "molecule evaluate"),
        ("330", "molecule generate"),
        ("60", "molecule verificate"),
        ("18", "molecule finale"),
    ]
    colors = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#FCE7F3"]

    for ax, (big, title), color in zip(axes.flat, cards, colors):
        ax.set_facecolor(color)
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=32, fontweight="bold")
        ax.text(0.5, 0.32, title, ha="center", va="center", fontsize=17, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "03_rezultate_pe_scurt.png", dpi=220)
    plt.close(fig)


def build_narrowing_chart(metrics: dict) -> None:
    ranked = metrics["ranked_molecules"]
    generated_total = (
        metrics["generated_candidate_count"]
        + metrics["ai_guided_candidate_count"]
        + metrics["iterative_candidate_count"]
    )
    docked = metrics["vina_docked_candidates"]
    final_batch = metrics["prospective_batch_size"]

    labels = [
        "Molecule evaluate",
        "Molecule generate",
        "Molecule verificate",
        "Lot final",
    ]
    values = [ranked, generated_total, docked, final_batch]
    colors = ["#60A5FA", "#34D399", "#FBBF24", "#F87171"]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_title("Cum se restrange spatiul chimic", fontweight="bold")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, ranked * 1.12)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ranked * 0.02,
            f"{value:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_restrangerea_spatiului_chimic.png", dpi=220)
    plt.close(fig)


def build_multi_agent_chart(rediscovery: dict) -> None:
    total_reference = rediscovery["positive_count"]
    found_simple = [
        round(rediscovery["naive_top10_recall"] * total_reference),
        round(rediscovery["naive_top20_recall"] * total_reference),
    ]
    found_multi = [
        round(rediscovery["protected_top10_recall"] * total_reference),
        round(rediscovery["protected_top20_recall"] * total_reference),
    ]
    missed_simple = [total_reference - v for v in found_simple]
    missed_multi = [total_reference - v for v in found_multi]

    categories = ["Simplu\nTop 10", "Simplu\nTop 20", "Multi-agent\nTop 10", "Multi-agent\nTop 20"]
    found = [found_simple[0], found_simple[1], found_multi[0], found_multi[1]]
    missed = [missed_simple[0], missed_simple[1], missed_multi[0], missed_multi[1]]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars_found = ax.bar(categories, found, color="#2DD4BF", width=0.62, label="Gasite")
    ax.bar(categories, missed, bottom=found, color="#E5E7EB", width=0.62, label="Ratate")

    ax.set_title("Cate molecule bune sunt gasite", fontweight="bold")
    ax.set_ylabel("Numar de molecule de referinta")
    ax.set_ylim(0, total_reference)
    ax.legend(frameon=False, loc="upper right")

    for bar, value in zip(bars_found, found):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6 if value > 0 else 0.6,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_multi_agent_gasite_vs_ratate.png", dpi=220)
    plt.close(fig)


def build_external_validation_chart(source_holdout: dict) -> None:
    labels_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
    }
    filtered = [item for item in source_holdout["results"] if item["source"] != "chembl"]
    labels = [labels_map[item["source"]] for item in filtered]
    values = [item["recall_top20pct"] * 100 for item in filtered]
    colors = ["#2563EB", "#0EA5E9", "#10B981"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_title("Validare pe baze de date externe", fontweight="bold")
    ax.set_ylabel("Molecule puternice gasite in primele 20% rezultate (%)")
    ax.set_ylim(0, 110)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_validare_pe_date_noi.png", dpi=220)
    plt.close(fig)


def build_generation_steps_chart(metrics: dict) -> None:
    labels = ["Generated", "AI-guided", "Iterative", "Lot final"]
    values = [
        metrics["generated_candidate_count"],
        metrics["ai_guided_candidate_count"],
        metrics["iterative_candidate_count"],
        metrics["prospective_batch_size"],
    ]
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171"]

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("Cate molecule au fost produse pe etape", fontweight="bold")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_molecule_generate_pe_etape.png", dpi=220)
    plt.close(fig)


def build_reference_panel_chart(rediscovery: dict) -> None:
    labels = ["Molecule bune", "Molecule challenger"]
    values = [rediscovery["positive_count"], rediscovery["challenger_count"]]
    colors = ["#22C55E", "#94A3B8"]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Setul de comparatie pentru testul final", fontweight="bold")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_setul_de_comparatie.png", dpi=220)
    plt.close(fig)


def build_final_batch_sources_chart(batch_summary: dict) -> None:
    source_map = {
        "diverse": "Diverse",
        "shortlist": "Shortlist",
        "optimized_readiness": "Optimized",
        "rl": "RL",
        "generated": "Generated",
    }
    source_counts = batch_summary["batch_sources"]
    labels = [source_map[key] for key in source_counts.keys()]
    values = list(source_counts.values())
    colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171", "#FBBF24"]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("De unde provin cele 18 molecule finale", fontweight="bold")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_surse_lot_final.png", dpi=220)
    plt.close(fig)


def build_final_batch_status_chart(batch_summary: dict) -> None:
    status_map = {
        "ready": "Ready",
        "supporting": "Supporting",
    }
    status_counts = batch_summary["batch_status_counts"]
    labels = [status_map[key] for key in status_counts.keys()]
    values = list(status_counts.values())
    colors = ["#22C55E", "#94A3B8"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title("Statusul lotului final", fontweight="bold")
    ax.set_ylabel("Numar de molecule")
    ax.set_ylim(0, max(values) * 1.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_status_lot_final.png", dpi=220)
    plt.close(fig)


def build_total_vs_unique_chart(multisource_summary: dict) -> None:
    labels = ["Total", "Unice"]
    values = [multisource_summary["interim_rows"], multisource_summary["unique_molecules"]]
    colors = ["#4F81BD", "#9BBB59"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Numarul de molecule", fontweight="bold")
    ax.set_ylabel("Numar")
    ax.set_ylim(0, max(values) * 1.1)
    ax.bar_label(bars, labels=[f"{value:,}".replace(",", ".") for value in values], padding=3, fontsize=12)

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "21_numar_total_si_unice.png", dpi=220)
    plt.close(fig)


def build_rmse_grade_chart(model_summary: dict) -> None:
    rmse_values = {
        "Random split": model_summary["random_split"]["rmse"],
        "Scaffold split": model_summary["scaffold_split"]["rmse"],
        "Temporal split": model_summary["temporal_split"]["rmse"],
    }
    rmse_min = min(rmse_values.values())
    rmse_max = max(rmse_values.values())
    grades = {
        label: max(1.0, min(10.0, 10 - 9 * (value - rmse_min) / (rmse_max - rmse_min)))
        for label, value in rmse_values.items()
    }

    labels = ["Random", "Scaffold", "Temporal"]
    values = list(grades.values())
    colors = ["#4F81BD", "#9BBB59", "#C0504D"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.58, label="Nota modelului")
    ax.plot([], [], " ", label="Formula: nota = 10 - 9 x (RMSE - min) / (max - min)")
    ax.set_title("Modelul ca nota", fontweight="bold")
    ax.set_ylabel("Nota")
    ax.set_ylim(0, 10.5)
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=12)

    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "22_rmse_ca_nota.png", dpi=220)
    plt.close(fig)


def build_best_vs_industry_chart(best_candidate: dict, marketed_reference: dict) -> None:
    labels = ["Putere", "Calitate", "Docking", "Interactii"]
    best_values = [
        best_candidate["predicted_pIC50"],
        best_candidate["QED"] * 10,
        best_candidate["docking_rescore"] * 10,
        best_candidate["interaction_support_score"] * 10,
    ]
    reference_values = [
        marketed_reference["predicted_pIC50"],
        marketed_reference["QED"] * 10,
        marketed_reference["docking_rescore"] * 10,
        marketed_reference["interaction_support_score"] * 10,
    ]

    positions = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar([p - width / 2 for p in positions], best_values, width=width, color="#4F81BD", label="Molecula noastra")
    ax.bar([p + width / 2 for p in positions], reference_values, width=width, color="#C0504D", label=marketed_reference["name"])
    ax.set_title("Molecula noastra vs standard", fontweight="bold")
    ax.set_ylabel("Scor 1-10")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "23_molecula_noastra_vs_standard.png", dpi=220)
    plt.close(fig)


def build_final_molecules_chart(batch_summary: dict) -> None:
    labels = ["Pool", "Finale", "Gata"]
    values = [
        batch_summary["candidate_pool_size"],
        batch_summary["selected_batch_size"],
        batch_summary["batch_status_counts"]["ready"],
    ]
    colors = ["#7F8FA6", "#4F81BD", "#9BBB59"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Cate molecule au ramas", fontweight="bold")
    ax.set_ylabel("Numar")
    ax.set_ylim(0, max(values) * 1.1)
    ax.bar_label(bars, labels=[str(value) for value in values], padding=3, fontsize=12)

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "24_cate_au_ramas.png", dpi=220)
    plt.close(fig)


def build_model_rmse_simple_chart(model_summary: dict) -> None:
    labels = ["Random", "Scaffold", "Temporal"]
    values = [
        model_summary["random_split"]["rmse"],
        model_summary["scaffold_split"]["rmse"],
        model_summary["temporal_split"]["rmse"],
    ]
    colors = ["#4F81BD", "#9BBB59", "#C0504D"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.58, label="Mai mic = mai bine")
    ax.set_title("Eroarea modelului", fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, max(values) * 1.2)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "25_eroarea_modelului.png", dpi=220)
    plt.close(fig)


def build_total_vs_unique_simple_chart(multisource_summary: dict) -> None:
    labels = ["Total", "Unice"]
    values = [multisource_summary["interim_rows"], multisource_summary["unique_molecules"]]
    colors = ["#4F81BD", "#9BBB59"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Total vs unice", fontweight="bold")
    ax.set_ylabel("Numar")
    ax.set_ylim(0, max(values) * 1.1)
    ax.bar_label(bars, labels=[f"{value:,}".replace(",", ".") for value in values], padding=3, fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "26_total_vs_unice.png", dpi=220)
    plt.close(fig)


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(REPORTS_DIR / "technical_notebook" / "technical_notebook_metrics.json")
    rediscovery = _load_json(REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_summary.json")
    source_holdout = _load_json(REPORTS_DIR / "source_holdout_benchmark.json")
    multisource_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "egfr_multisource_summary.json")
    batch_summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    model_summary = _load_json(REPORTS_DIR / "model_performance_summary.json")
    best_candidate = _load_csv_rows(REPORTS_DIR / "prospective_validation_batch.csv")[0]
    marketed_rows = _load_csv_rows(REPORTS_DIR / "marketed_egfr_structural_benchmark.csv")
    marketed_reference = next(
        row for row in marketed_rows if row.get("name", "").strip().lower() == "osimertinib"
    )

    build_context_cards()
    build_public_sources_chart(multisource_summary)
    build_simple_pipeline()
    build_results_summary(metrics)
    build_narrowing_chart(metrics)
    build_multi_agent_chart(rediscovery)
    build_external_validation_chart(source_holdout)
    build_generation_steps_chart(metrics)
    build_reference_panel_chart(rediscovery)
    build_final_batch_sources_chart(batch_summary)
    build_final_batch_status_chart(batch_summary)
    build_total_vs_unique_chart(multisource_summary)
    build_rmse_grade_chart(model_summary)
    build_best_vs_industry_chart(
        {
            "predicted_pIC50": float(best_candidate["predicted_pIC50"]),
            "QED": float(best_candidate["QED"]),
            "docking_rescore": float(best_candidate["docking_rescore"]),
            "interaction_support_score": float(best_candidate["interaction_support_score"]),
        },
        {
            "name": marketed_reference["name"],
            "predicted_pIC50": float(marketed_reference["predicted_pIC50"]),
            "QED": float(marketed_reference["QED"]),
            "docking_rescore": float(marketed_reference["docking_rescore"]),
            "interaction_support_score": float(marketed_reference["interaction_support_score"]),
        },
    )
    build_final_molecules_chart(batch_summary)
    build_model_rmse_simple_chart(model_summary)
    build_total_vs_unique_simple_chart(multisource_summary)

    print(f"[OK] Saved simplified presentation visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
