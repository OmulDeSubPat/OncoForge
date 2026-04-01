from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "presentation_visuals_davinci"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


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
            "axes.titlesize": 22,
            "axes.labelsize": 13,
            "font.family": "DejaVu Sans",
        }
    )


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = "#1F2937",
    linewidth: float = 1.8,
    radius: float = 0.04,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    return patch


def build_problem_chart(metrics: dict, batch_summary: dict) -> None:
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.8),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.suptitle("Problema abordata", fontsize=24, fontweight="bold", y=0.98)

    labels = ["Molecule evaluate", "Pool selectat", "Lot final"]
    values = [
        metrics["ranked_molecules"],
        batch_summary["candidate_pool_size"],
        batch_summary["selected_batch_size"],
    ]
    colors = ["#60A5FA", "#F59E0B", "#EF4444"]
    positions = [2, 1, 0]

    ax_left.barh(positions, values, color=colors, height=0.58)
    ax_left.set_yticks(positions)
    ax_left.set_yticklabels(labels)
    ax_left.set_xlabel("Numar de molecule")
    ax_left.set_xlim(0, values[0] * 1.12)
    ax_left.set_title("Din mii de optiuni, foarte putine pot merge mai departe", fontsize=16, pad=10)

    for y, value in zip(positions, values):
        ax_left.text(
            value + values[0] * 0.015,
            y,
            _format_int(value),
            va="center",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

    ax_left.grid(axis="x", alpha=0.25)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    ax_right.axis("off")
    _rounded_box(ax_right, 0.06, 0.68, 0.88, 0.22, "#DBEAFE")
    _rounded_box(ax_right, 0.06, 0.39, 0.88, 0.22, "#FEF3C7")
    _rounded_box(ax_right, 0.06, 0.10, 0.88, 0.22, "#DCFCE7")

    ax_right.text(
        0.10,
        0.79,
        "1. Spatiul chimic este foarte mare,\nasa ca nu putem testa totul experimental.",
        fontsize=14,
        fontweight="bold",
        va="center",
    )
    ax_right.text(
        0.10,
        0.50,
        "2. Validarea in laborator este lenta\nsi costisitoare pentru fiecare candidat.",
        fontsize=14,
        fontweight="bold",
        va="center",
    )
    ax_right.text(
        0.10,
        0.21,
        "3. Avem nevoie de o selectie automata,\ndar si de filtre de siguranta si realism.",
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "01_problema_abordata.png", dpi=220)
    plt.close(fig)


def build_scientific_context_chart() -> None:
    fig = plt.figure(figsize=(14.5, 7.2))
    fig.suptitle("Context stiintific", fontsize=24, fontweight="bold", y=0.98)

    ax_cards = fig.add_axes([0.05, 0.30, 0.90, 0.56])
    ax_cards.axis("off")

    cards = [
        ("2.480.675", "cazuri noi de cancer pulmonar", "#DBEAFE"),
        ("717.211", "cazuri de adenocarcinom la barbati", "#DCFCE7"),
        ("541.971", "cazuri de adenocarcinom la femei", "#FEF3C7"),
    ]

    for idx, (big, subtitle, color) in enumerate(cards):
        x = 0.01 + idx * 0.33
        _rounded_box(ax_cards, x, 0.10, 0.30, 0.78, color, linewidth=0.0, radius=0.04)
        ax_cards.text(x + 0.15, 0.58, big, ha="center", va="center", fontsize=27, fontweight="bold")
        ax_cards.text(x + 0.15, 0.32, subtitle, ha="center", va="center", fontsize=14, wrap=True)

    ax_note = fig.add_axes([0.08, 0.08, 0.84, 0.14])
    ax_note.axis("off")
    _rounded_box(ax_note, 0.0, 0.0, 1.0, 1.0, "#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5, radius=0.035)
    ax_note.text(
        0.5,
        0.62,
        "EGFR este o tinta moleculara importanta in oncologie, mai ales in subtipuri de cancer pulmonar.",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax_note.text(
        0.5,
        0.24,
        "Sursa pentru datele epidemiologice: IARC GLOBOCAN 2022.",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    fig.savefig(OUTPUT_DIR / "02_context_stiintific.png", dpi=220)
    plt.close(fig)


def build_abstract_chart(model_summary: dict, metrics: dict, batch_summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Abstractul proiectului", fontsize=24, fontweight="bold", y=0.97)

    boxes = [
        (
            0.02,
            "#DBEAFE",
            "Input",
            f"{_format_int(model_summary['dataset_size'])} molecule\nmultisursa pentru model",
        ),
        (
            0.27,
            "#DCFCE7",
            "Metoda",
            "Scorare multi-agent\n+ reward verificabil",
        ),
        (
            0.52,
            "#FEF3C7",
            "Validare",
            f"Scaffold RMSE {model_summary['scaffold_split']['rmse']:.3f}\nRecall extern top 20%: {metrics['source_holdout_mean_recall_top20pct'] * 100:.0f}%",
        ),
        (
            0.77,
            "#FCE7F3",
            "Output",
            f"{batch_summary['selected_batch_size']} candidati finali\n{batch_summary['batch_status_counts']['ready']} ready",
        ),
    ]

    for index, (x, color, title, body) in enumerate(boxes):
        _rounded_box(ax, x, 0.26, 0.20, 0.46, color, radius=0.03)
        ax.text(x + 0.10, 0.61, title, ha="center", va="center", fontsize=16, fontweight="bold")
        ax.text(x + 0.10, 0.42, body, ha="center", va="center", fontsize=13.2)
        if index < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.01, 0.49),
                xytext=(x + 0.20, 0.49),
                arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#475569"},
            )

    ax.text(
        0.5,
        0.10,
        "Ideea centrala: sistemul nu pretinde descoperirea unui medicament final, ci prioritizeaza candidati EGFR mai buni pentru validare ulterioara.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#334155",
    )

    fig.tight_layout(rect=(0.01, 0, 0.99, 0.93))
    fig.savefig(OUTPUT_DIR / "03_abstract_proiect.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def build_data_chart(multisource_summary: dict) -> None:
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.8),
        gridspec_kw={"width_ratios": [1.3, 0.9]},
    )
    fig.suptitle("Datele proiectului", fontsize=24, fontweight="bold", y=0.98)

    labels = ["ChEMBL", "Papyrus", "ExCAPE", "BindingDB"]
    values = [
        multisource_summary["chembl_rows"],
        multisource_summary["papyrus_rows"],
        multisource_summary["excape_rows"],
        multisource_summary["bindingdb_rows"],
    ]
    colors = ["#2563EB", "#0EA5E9", "#10B981", "#F59E0B"]

    bars = ax_left.bar(labels, values, color=colors, width=0.62)
    ax_left.set_title("Sursele publice integrate", fontsize=16, pad=10)
    ax_left.set_ylabel("Numar de inregistrari")
    ax_left.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            _format_int(value),
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
        )

    ax_left.grid(axis="y", alpha=0.25)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    ax_right.axis("off")
    _rounded_box(ax_right, 0.08, 0.54, 0.84, 0.28, "#DBEAFE")
    _rounded_box(ax_right, 0.08, 0.16, 0.84, 0.28, "#DCFCE7")

    ax_right.text(
        0.50,
        0.68,
        _format_int(multisource_summary["unique_molecules"]),
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    ax_right.text(
        0.50,
        0.57,
        "molecule unice dupa curatare",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax_right.text(
        0.50,
        0.30,
        _format_int(multisource_summary["molecules_with_multiple_sources"]),
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    ax_right.text(
        0.50,
        0.19,
        "molecule sustinute din mai multe surse",
        ha="center",
        va="center",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "04_datele_proiectului.png", dpi=220)
    plt.close(fig)


def build_pipeline_chart() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Cum lucreaza sistemul", fontsize=24, fontweight="bold", y=0.96)

    boxes = [
        (0.02, "#DBEAFE", "1. Date\npublice"),
        (0.18, "#D1FAE5", "2. Model\nensemble"),
        (0.34, "#FEF3C7", "3. Scorare\nmulti-agent"),
        (0.50, "#FCE7F3", "4. Generare\nsi filtre"),
        (0.66, "#EDE9FE", "5. Validare\nexterna"),
        (0.82, "#FEE2E2", "6. Lot\nfinal"),
    ]

    width = 0.13
    height = 0.34
    y = 0.52

    for index, (x, color, text) in enumerate(boxes):
        _rounded_box(ax, x, y - height / 2, width, height, color, radius=0.025)
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=14, fontweight="bold")
        if index < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.008, y),
                xytext=(x + width, y),
                arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": "#475569"},
            )

    ax.text(
        0.50,
        0.11,
        "Fiecare etapa elimina candidatii slabi si pastreaza moleculele mai plauzibile pentru EGFR.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#475569",
    )

    fig.tight_layout(rect=(0.01, 0, 0.99, 0.92))
    fig.savefig(OUTPUT_DIR / "05_pipeline_oncoforge.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def build_results_chart(metrics: dict, batch_summary: dict, model_summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.2))
    fig.suptitle("Rezultate cheie", fontsize=24, fontweight="bold", y=0.98)

    cards = [
        (_format_int(metrics["ranked_molecules"]), "molecule evaluate", "#DBEAFE"),
        (f"{model_summary['scaffold_split']['rmse']:.3f}", "RMSE pe scaffold split", "#DCFCE7"),
        (f"{metrics['source_holdout_mean_recall_top20pct'] * 100:.0f}%", "recall mediu pe surse externe", "#FEF3C7"),
        (str(batch_summary["selected_batch_size"]), "candidati in lotul final", "#FCE7F3"),
    ]

    for ax, (big, subtitle, color) in zip(axes.flat, cards):
        ax.set_facecolor(color)
        ax.text(0.5, 0.60, big, ha="center", va="center", fontsize=30, fontweight="bold")
        ax.text(0.5, 0.32, subtitle, ha="center", va="center", fontsize=15, fontweight="bold", wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "06_rezultate_cheie.png", dpi=220)
    plt.close(fig)


def build_multi_agent_chart(metrics: dict, rediscovery: dict) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.6, 6.8))
    fig.suptitle("De ce conteaza multi-agent + audit", fontsize=24, fontweight="bold", y=0.98)

    categories = ["Top 10", "Top 20"]
    naive = [
        rediscovery["naive_top10_recall"] * 100,
        rediscovery["naive_top20_recall"] * 100,
    ]
    protected = [
        rediscovery["protected_top10_recall"] * 100,
        rediscovery["protected_top20_recall"] * 100,
    ]
    x = [0, 1]
    width = 0.34

    ax_left.bar([value - width / 2 for value in x], naive, width=width, color="#94A3B8", label="Scor simplu")
    ax_left.bar([value + width / 2 for value in x], protected, width=width, color="#10B981", label="Multi-agent")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(categories)
    ax_left.set_ylim(0, 50)
    ax_left.set_ylabel("Recall pentru molecule de referinta (%)")
    ax_left.set_title("Rediscovery benchmark")
    ax_left.legend(frameon=False, loc="upper left")
    for index, value in enumerate(naive):
        ax_left.text(index - width / 2, value + 1.2, f"{value:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for index, value in enumerate(protected):
        ax_left.text(index + width / 2, value + 1.2, f"{value:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    right_labels = ["Trusted\npassed", "Proxy exploits\ndemoted", "Proxy exploits\nreview/fail"]
    right_values = [
        metrics["challenge_trusted_pass_rate"] * 100,
        metrics["challenge_proxy_demoted_rate"] * 100,
        metrics["challenge_proxy_review_or_fail_rate"] * 100,
    ]
    right_colors = ["#22C55E", "#EF4444", "#F59E0B"]

    bars = ax_right.bar(right_labels, right_values, color=right_colors, width=0.58)
    ax_right.set_ylim(0, 110)
    ax_right.set_ylabel("Rata (%)")
    ax_right.set_title("Reward-hacking challenge")
    for bar, value in zip(bars, right_values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "07_multi_agent_si_audit.png", dpi=220)
    plt.close(fig)


def build_external_validation_chart(source_holdout: dict) -> None:
    source_map = {
        "excape_chembl20": "ExCAPE",
        "papyrus": "Papyrus",
        "bindingdb_articles": "BindingDB",
    }
    filtered = [item for item in source_holdout["results"] if item["source"] != "chembl"]

    labels = [source_map[item["source"]] for item in filtered]
    recall_values = [item["recall_top20pct"] * 100 for item in filtered]
    rmse_values = [item["rmse"] for item in filtered]
    colors = ["#2563EB", "#0EA5E9", "#10B981"]

    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    bars = ax.bar(labels, recall_values, color=colors, width=0.62)
    ax.set_title("Validare pe date noi", fontweight="bold")
    ax.set_ylabel("Recall in top 20% (%)")
    ax.set_ylim(0, 110)

    for bar, recall, rmse in zip(bars, recall_values, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            recall + 2.2,
            f"{recall:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(10, recall * 0.40),
            f"RMSE {rmse:.2f}",
            ha="center",
            va="center",
            fontsize=11.5,
            color="white",
            fontweight="bold",
        )

    ax.text(
        0.01,
        -0.13,
        "Interpretare: modelul recupereaza bine molecule puternice si pe surse externe, nu doar pe datele folosite la antrenare.",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#475569",
    )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_validare_pe_date_noi.png", dpi=220)
    plt.close(fig)


def build_final_batch_chart(batch_summary: dict) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 6.6))
    fig.suptitle("Lotul final pentru validare", fontsize=24, fontweight="bold", y=0.98)

    source_map = {
        "diverse": "Diverse",
        "shortlist": "Shortlist",
        "optimized_readiness": "Optimized",
        "rl": "RL",
        "generated": "Generated",
    }
    source_counts = batch_summary["batch_sources"]
    left_labels = [source_map[key] for key in source_counts.keys()]
    left_values = list(source_counts.values())
    left_colors = ["#60A5FA", "#34D399", "#A78BFA", "#F87171", "#FBBF24"]

    bars_left = ax_left.bar(left_labels, left_values, color=left_colors, width=0.62)
    ax_left.set_title("Din ce etape provin cele 18 molecule")
    ax_left.set_ylabel("Numar de molecule")
    ax_left.set_ylim(0, max(left_values) * 1.28)
    for bar, value in zip(bars_left, left_values):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    status_labels = ["Ready", "Supporting"]
    status_values = [
        batch_summary["batch_status_counts"]["ready"],
        batch_summary["batch_status_counts"]["supporting"],
    ]
    status_colors = ["#22C55E", "#94A3B8"]

    bars_right = ax_right.bar(status_labels, status_values, color=status_colors, width=0.56)
    ax_right.set_title("Statusul lotului final")
    ax_right.set_ylabel("Numar de molecule")
    ax_right.set_ylim(0, max(status_values) * 1.35)
    for bar, value in zip(bars_right, status_values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax_right.text(
        0.5,
        0.08,
        f"Mean readiness score: {batch_summary['mean_readiness_score']:.3f}",
        transform=ax_right.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "09_lotul_final.png", dpi=220)
    plt.close(fig)


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(REPORTS_DIR / "technical_notebook" / "technical_notebook_metrics.json")
    multisource_summary = _load_json(PROJECT_ROOT / "data" / "processed" / "egfr_multisource_summary.json")
    batch_summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    model_summary = _load_json(REPORTS_DIR / "model_performance_summary.json")
    rediscovery = _load_json(REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_summary.json")
    source_holdout = _load_json(REPORTS_DIR / "source_holdout_benchmark.json")

    build_problem_chart(metrics, batch_summary)
    build_scientific_context_chart()
    build_abstract_chart(model_summary, metrics, batch_summary)
    build_data_chart(multisource_summary)
    build_pipeline_chart()
    build_results_chart(metrics, batch_summary, model_summary)
    build_multi_agent_chart(metrics, rediscovery)
    build_external_validation_chart(source_holdout)
    build_final_batch_chart(batch_summary)

    print(f"[OK] Saved Da Vinci presentation visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
