from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from src.config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "presentation_visuals_davinci_v3"


PROBLEM_BURDEN_DATA = [
    {
        "year": 2018,
        "cases_millions": 2.10,
        "deaths_millions": 1.76,
        "source": "Bray et al., Global cancer statistics 2018, PubMed PMID 30207593.",
    },
    {
        "year": 2020,
        "cases_millions": 2.206771,
        "deaths_millions": 1.796144,
        "source": "Sung et al., Global Cancer Statistics 2020, PubMed PMID 33538338.",
    },
    {
        "year": 2022,
        "cases_millions": 2.480301,
        "deaths_millions": 1.820000,
        "source": "IARC / GLOBOCAN 2022 lung cancer summary.",
    },
    {
        "year": 2050,
        "cases_millions": 4.62,
        "deaths_millions": 3.55,
        "source": "Zhou et al., Global burden of lung cancer in 2022 and projections to 2050, Cancer Epidemiology, 2024.",
    },
]

PUBMED_MOLECULE_GENERATION_QUERY = (
    '("de novo molecular design"[Title/Abstract] OR "generative molecular design"[Title/Abstract] '
    'OR "molecule generation"[Title/Abstract] OR "molecular generation"[Title/Abstract])'
)

PUBMED_MOLECULE_GENERATION_COUNTS = {
    2016: 8,
    2017: 6,
    2018: 13,
    2019: 13,
    2020: 25,
    2021: 54,
    2022: 65,
    2023: 65,
    2024: 123,
    2025: 123,
}

DRUG_APPROVAL_YEARS = {
    "Gefitinib": 2003,
    "Erlotinib": 2004,
    "Afatinib": 2013,
    "Osimertinib": 2015,
    "Dacomitinib": 2018,
    "Sunvozertinib": 2023,
}

CONTEXT_MARKET_ROWS = [
    ("Gefitinib", "2003", "tratament aprobat", "mutatii activatoare"),
    ("Erlotinib", "2004", "tratament aprobat", "mutatii activatoare"),
    ("Afatinib", "2013", "tratament aprobat", "familia ErbB"),
    ("Osimertinib", "2015", "tratament aprobat", "T790M / rezistenta"),
    ("Dacomitinib", "2018", "tratament aprobat", "mutatii activatoare"),
    ("Sunvozertinib", "2023", "tratament aprobat", "exon 20"),
    ("OncoSynth", "2026", "platforma AI", "candidati EGFR noi"),
]

STUDY_TABLE_ROWS = [
    {
        "study": "OncoSynth",
        "task": "set multisursa EGFR",
        "metric": "Numar molecule",
        "value": "10.606",
        "highlight": True,
    },
    {
        "study": "Nada et al.\n2023",
        "task": "set EGFR curat",
        "metric": "Numar molecule",
        "value": "~9.000",
        "highlight": False,
    },
    {
        "study": "DeepEGFR\n2025",
        "task": "set EGFR clasificare",
        "metric": "Numar molecule",
        "value": "8.263",
        "highlight": False,
    },
    {
        "study": "OncoSynth",
        "task": "regresie pIC50,\nsplit aleator",
        "metric": "R^2",
        "value": "0.747",
        "highlight": True,
    },
    {
        "study": "Nada et al.\n2023",
        "task": "regresie pIC50,\nvalidare 5-fold",
        "metric": "R^2",
        "value": "0.717",
        "highlight": False,
    },
    {
        "study": "OncoSynth",
        "task": "regresie pIC50,\nschelete diferite",
        "metric": "R^2",
        "value": "0.683",
        "highlight": True,
    },
    {
        "study": "Nada et al.\n2023",
        "task": "subgrup specializat,\nschelete",
        "metric": "R^2",
        "value": "0.860",
        "highlight": False,
    },
    {
        "study": "DeepEGFR\n2025",
        "task": "clasificare EGFR,\nschelete",
        "metric": "F1",
        "value": "0.940",
        "highlight": False,
    },
]

CONTEXT_REFERENCE_LINES = [
    "Bray F et al. Global cancer statistics 2018. CA Cancer J Clin. 2018.",
    "Sung H et al. Global Cancer Statistics 2020. CA Cancer J Clin. 2021.",
    "IARC. Lung cancer global summary / GLOBOCAN 2022.",
    "Zhou J et al. Global burden of lung cancer in 2022 and projections to 2050. Cancer Epidemiology. 2024.",
]

SOURCE_LINKS = [
    "- Global cancer statistics 2018: https://pubmed.ncbi.nlm.nih.gov/30207593/",
    "- Global cancer statistics 2020: https://pubmed.ncbi.nlm.nih.gov/33538338/",
    "- IARC lung cancer page: https://www.iarc.who.int/cancer-type/lung-cancer/",
    "- Lung cancer 2022 to 2050 projections: https://www.sciencedirect.com/science/article/pii/S1877782124001723",
    "- Gefitinib first major approval context: https://pubmed.ncbi.nlm.nih.gov/12897327/",
    "- Erlotinib FDA approval summary: https://pubmed.ncbi.nlm.nih.gov/16079312/",
    "- Afatinib FDA broadening note with original 2013 approval: https://www.fda.gov/drugs/resources-information-approved-drugs/fda-broadens-afatinib-indication-previously-untreated-metastatic-nsclc-other-non-resistant-egfr",
    "- Osimertinib FDA approval summary: https://www.fda.gov/drugs/resources-information-approved-drugs/osimertinib-tagrisso",
    "- Dacomitinib FDA approval summary: https://www.fda.gov/drugs/drug-approvals-and-databases/fda-approves-dacomitinib-metastatic-non-small-cell-lung-cancer-0",
    "- Sunvozertinib FDA approval summary (2025): https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-accelerated-approval-sunvozertinib-metastatic-non-small-cell-lung-cancer-egfr-exon-20",
    "- PubMed query used for publication trend: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_int(value: int | float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def _build_reference_lines() -> list[str]:
    rows = _load_csv_rows(REPORTS_DIR / "technical_notebook" / "reference_library.csv")
    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        citation = (row.get("citation") or "").strip()
        title = (row.get("title") or "").strip()
        if citation and title:
            lines.append(f"[{idx}] {citation} {title}.")
        elif citation:
            lines.append(f"[{idx}] {citation}")
        elif title:
            lines.append(f"[{idx}] {title}")

    offset = len(lines)
    for extra_idx, context_line in enumerate(CONTEXT_REFERENCE_LINES, start=1):
        lines.append(f"[{offset + extra_idx}] {context_line}")
    return lines


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0F172A",
            "font.size": 12,
            "axes.titlesize": 20,
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
    edgecolor: str = "#0F172A",
    linewidth: float = 1.6,
    radius: float = 0.02,
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


def build_problem_chart() -> None:
    anchor_2022 = next(item for item in PROBLEM_BURDEN_DATA if item["year"] == 2022)
    anchor_2050 = next(item for item in PROBLEM_BURDEN_DATA if item["year"] == 2050)
    progress_to_2030 = (2030 - 2022) / (2050 - 2022)
    projected_2030_cases = anchor_2022["cases_millions"] + (
        anchor_2050["cases_millions"] - anchor_2022["cases_millions"]
    ) * progress_to_2030
    projected_2030_deaths = anchor_2022["deaths_millions"] + (
        anchor_2050["deaths_millions"] - anchor_2022["deaths_millions"]
    ) * progress_to_2030

    years = [2018, 2020, 2022, 2030]
    cases = [
        PROBLEM_BURDEN_DATA[0]["cases_millions"],
        PROBLEM_BURDEN_DATA[1]["cases_millions"],
        PROBLEM_BURDEN_DATA[2]["cases_millions"],
        projected_2030_cases,
    ]
    deaths = [
        PROBLEM_BURDEN_DATA[0]["deaths_millions"],
        PROBLEM_BURDEN_DATA[1]["deaths_millions"],
        PROBLEM_BURDEN_DATA[2]["deaths_millions"],
        projected_2030_deaths,
    ]

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.plot(years, cases, color="#0F766E", linewidth=3.4)
    ax.plot(years, deaths, color="#B91C1C", linewidth=3.4, linestyle="--")

    ax.set_xlabel("Ani")
    ax.set_ylabel("Cazuri / Morti (milioane)")
    ax.set_xlim(2018, 2030)
    ax.set_xticks([2018, 2020, 2022, 2024, 2026, 2028, 2030])
    ax.set_ylim(1.5, 3.4)
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_problema_abordata_burden.png", dpi=240)
    plt.close(fig)


def build_market_table_chart() -> None:
    table_rows = [list(row) for row in CONTEXT_MARKET_ROWS]
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.axis("off")

    table = ax.table(
        cellText=table_rows,
        colLabels=["Entitate", "An", "Tip", "Context"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.04, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.8)
    table.scale(1, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row_idx == 0:
            cell.set_facecolor("#DBEAFE")
            cell.set_text_props(fontweight="bold")
        elif table_rows[row_idx - 1][0] == "OncoSynth":
            cell.set_facecolor("#DCFCE7")
            if col_idx in {0, 2}:
                cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 else "#EEF2FF")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03a_tabel_medicamente.png", dpi=240)
    plt.close(fig)


def build_pubmed_trend_chart() -> None:
    years = list(PUBMED_MOLECULE_GENERATION_COUNTS.keys())
    counts = list(PUBMED_MOLECULE_GENERATION_COUNTS.values())

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.plot(years, counts, color="#F59E0B", linewidth=3.2, marker="o", markersize=6)
    ax.fill_between(years, counts, color="#FDE68A", alpha=0.35)
    ax.set_xlabel("Ani")
    ax.set_ylabel("Articole pe an")
    ax.set_xticks(years)
    ax.set_xlim(min(years), max(years))
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03b_trend_publicatii.png", dpi=240)
    plt.close(fig)


def build_chemical_space_chart(metrics: dict, batch_summary: dict) -> None:
    stages = [
        ("Spatiu teoretic", 1e60),
        ("Set multisursa", metrics["ranked_molecules"]),
        (
            "Molecule generate",
            metrics["generated_candidate_count"]
            + metrics["ai_guided_candidate_count"]
            + metrics["iterative_candidate_count"],
        ),
        ("Pool selectie", batch_summary["candidate_pool_size"]),
        ("Candidati docati", metrics["vina_docked_candidates"]),
        ("Lot final", batch_summary["selected_batch_size"]),
    ]

    labels = [item[0] for item in stages]
    values = [item[1] for item in stages]
    log_values = [60.0] + [math.log10(value) for value in values[1:]]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(range(len(labels)), log_values, color="#2563EB", linewidth=3.2, marker="o", markersize=8)
    ax.fill_between(range(len(labels)), log_values, color="#BFDBFE", alpha=0.35)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Numar de molecule (scara logaritmica)")
    ax.set_xlabel("Etape")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(values):
        label = "10^60" if value >= 1e50 else _format_int(value)
        ax.text(idx, log_values[idx] + 1.0, label, ha="center", va="bottom", fontsize=11.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_abstract_spatiu_chimic.png", dpi=240)
    plt.close(fig)


def build_methodology_chart(model_summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.03, "#DBEAFE", "Colectare si\nfiltrare de date"),
        (0.22, "#DCFCE7", "Antrenarea mai\nmultor modele"),
        (0.41, "#FEF3C7", "Generarea,\ntestarea si\nfiltrarea"),
        (0.60, "#FCE7F3", "Validarea\ncandidatilor"),
        (0.79, "#FEE2E2", "Alegerea\nlotului final"),
    ]
    width = 0.14
    height = 0.34
    y = 0.58

    for idx, (x, color, label) in enumerate(steps):
        _rounded_box(ax, x, y - height / 2, width, height, color, radius=0.03)
        ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=14.2, fontweight="bold")
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(steps[idx + 1][0] - 0.01, y),
                xytext=(x + width, y),
                arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "#475569"},
            )

    _rounded_box(ax, 0.63, 0.08, 0.30, 0.22, "#F8FAFC", edgecolor="#94A3B8", linewidth=1.8, radius=0.03)
    ax.text(0.78, 0.23, "Cel mai bun model", ha="center", va="center", fontsize=14.5, fontweight="bold")
    ax.text(0.78, 0.16, "Ansamblu combinat", ha="center", va="center", fontsize=14)
    ax.text(
        0.78,
        0.10,
        f"RMSE pe schelete = {model_summary['scaffold_split']['rmse']:.3f}   |   R^2 = {model_summary['scaffold_split']['r2']:.3f}",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#1E3A8A",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_metodologie.png", dpi=240, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _model_rows(model_summary: dict, family_benchmark: dict) -> list[dict]:
    label_map = {
        "et_ecfp": "Arbori extra + amprente",
        "hgb_ecfp": "Boosting + amprente",
        "rf_hybrid": "Padure aleatoare + mixt",
        "mlp_hybrid": "Retea neuronala + mixt",
        "rf_descriptors": "Padure + descriptori",
        "knn_descriptors": "Vecini + descriptori",
        "ridge_descriptors": "Regresie penalizata + descriptori",
    }
    rows = []
    for row in family_benchmark["results"]:
        rows.append(
            {
                "label": label_map.get(row["model_family"], row["model_family"]),
                "rmse": row["rmse_scaffold"],
                "r2": row["r2_scaffold"],
                "color": "#2563EB" if row["feature_set"] == "ecfp" else "#10B981" if row["feature_set"] == "hybrid" else "#F59E0B",
            }
        )
    rows.append(
        {
            "label": "Ansamblu final",
            "rmse": model_summary["scaffold_split"]["rmse"],
            "r2": model_summary["scaffold_split"]["r2"],
            "color": "#DC2626",
        }
    )
    rows.sort(key=lambda item: item["rmse"])
    return rows


def build_model_rmse_chart(model_summary: dict, family_benchmark: dict) -> None:
    rows = _model_rows(model_summary, family_benchmark)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    labels = [row["label"] for row in rows]
    values = [row["rmse"] for row in rows]
    colors = [row["color"] for row in rows]
    bars = ax.barh(labels, values, color=colors, height=0.68)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE pe validare cu schelete diferite")
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06a_rmse_modele.png", dpi=240)
    plt.close(fig)


def build_model_r2_chart(model_summary: dict, family_benchmark: dict) -> None:
    rows = _model_rows(model_summary, family_benchmark)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    labels = [row["label"] for row in rows]
    values = [row["r2"] for row in rows]
    colors = [row["color"] for row in rows]
    bars = ax.barh(labels, values, color=colors, height=0.68)
    ax.invert_yaxis()
    ax.set_xlabel("R^2 pe validare cu schelete diferite")
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06b_r2_modele.png", dpi=240)
    plt.close(fig)


def build_multi_agent_chart(metrics: dict, rediscovery: dict) -> None:
    fig = plt.figure(figsize=(15.6, 7.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    labels = ["Primele 10", "Primele 20"]
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
    ax_a.bar([value - width / 2 for value in x], naive, width=width, color="#94A3B8", label="Naiv")
    ax_a.bar([value + width / 2 for value in x], protected, width=width, color="#10B981", label="Protejat")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, 45)
    ax_a.set_title("Regasire de molecule cunoscute", fontsize=16)
    ax_a.set_ylabel("Rata de regasire (%)")
    ax_a.legend(frameon=False, loc="upper left")
    ax_a.grid(axis="y", alpha=0.18)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    challenge_labels = ["Cazuri curate", "Cazuri inselatoare\ncoborate 20+", "Cazuri inselatoare\nla revizie/respins"]
    challenge_values = [
        metrics["challenge_trusted_pass_rate"] * 100,
        metrics["challenge_proxy_demoted_rate"] * 100,
        metrics["challenge_proxy_review_or_fail_rate"] * 100,
    ]
    challenge_colors = ["#22C55E", "#DC2626", "#F59E0B"]
    bars = ax_b.bar(challenge_labels, challenge_values, color=challenge_colors, width=0.6)
    ax_b.set_ylim(0, 110)
    ax_b.set_title("Test anti-manipulare a scorului", fontsize=16)
    ax_b.set_ylabel("Rata (%)")
    for bar, value in zip(bars, challenge_values):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax_b.grid(axis="y", alpha=0.18)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    status_counts = metrics["status_counts"]
    pie_labels = ["Acceptat", "Revizie", "Respins"]
    pie_values = [status_counts["pass"], status_counts["review"], status_counts["fail"]]
    pie_colors = ["#16A34A", "#F59E0B", "#DC2626"]
    ax_c.pie(
        pie_values,
        labels=pie_labels,
        autopct=lambda pct: f"{pct:.0f}%",
        startangle=90,
        colors=pie_colors,
        textprops={"color": "#0F172A", "fontsize": 11},
    )
    ax_c.set_title("Audit pe 16.133 molecule", fontsize=16)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_rezultate_multi_agent.png", dpi=240)
    plt.close(fig)


def build_ai_vs_studies_chart() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 7.8))
    ax.axis("off")
    ax.text(
        0.0,
        1.04,
        "Comparatii cu studii EGFR",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
    )

    cell_text = [
        [row["study"], row["task"], row["metric"], row["value"]]
        for row in STUDY_TABLE_ROWS
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Studiu", "Setare", "Metrica", "Valoare"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.08, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row_idx == 0:
            cell.set_facecolor("#DBEAFE")
            cell.set_text_props(fontweight="bold")
        else:
            highlight = STUDY_TABLE_ROWS[row_idx - 1]["highlight"]
            if highlight:
                cell.set_facecolor("#DCFCE7")
                if col_idx in {0, 2, 3}:
                    cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#F8FAFC" if row_idx % 2 else "#EEF2FF")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_ai_vs_studii.png", dpi=240)
    plt.close(fig)


def build_references_chart() -> None:
    reference_lines = _build_reference_lines()
    fig, ax = plt.subplots(figsize=(15.4, 9.2))
    ax.axis("off")
    ax.text(0.00, 1.02, "Referinte", transform=ax.transAxes, fontsize=21, fontweight="bold", va="bottom")

    midpoint = (len(reference_lines) + 1) // 2
    left_lines = reference_lines[:midpoint]
    right_lines = reference_lines[midpoint:]

    def draw_column(x: float, lines: list[str]) -> None:
        y = 0.95
        for line in lines:
            wrapped = textwrap.fill(line, width=54)
            ax.text(x, y, wrapped, transform=ax.transAxes, fontsize=9.8, va="top")
            y -= 0.085 + 0.010 * wrapped.count("\n")

    draw_column(0.00, left_lines)
    draw_column(0.51, right_lines)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_referinte.png", dpi=240, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def build_title_template() -> None:
    lines = [
        "Titlu proiect: OncoForge - Platforma AI pentru prioritizarea moleculelor EGFR",
        "Autori: [Completeaza numele autorilor]",
        "Coordonator: [Completeaza profesorul coordonator]",
        "Competitie: [Completeaza sectiunea / anul]",
    ]
    (OUTPUT_DIR / "01_titlu_autori_template.txt").write_text("\n".join(lines), encoding="utf-8")


def build_notes_file() -> None:
    notes = [
        "# Da Vinci Slide Graphics",
        "",
        "Slide 1: 01_titlu_autori_template.txt",
        "Slide 2: 02_problema_abordata_burden.png",
        "Slide 3A: 03a_tabel_medicamente.png",
        "Slide 3B: 03b_trend_publicatii.png",
        "Slide 4: 04_abstract_spatiu_chimic.png",
        "Slide 5: 05_metodologie.png",
        "Slide 6A: 06a_rmse_modele.png",
        "Slide 6B: 06b_r2_modele.png",
        "Slide 7: 07_rezultate_multi_agent.png",
        "Slide 8: 08_ai_vs_studii.png",
        "Slide 9: planuri de viitor - utilizatorul completeaza manual.",
        "Slide 10: 10_referinte.png",
        "",
        "## Important",
        "- Slide-ul 2 foloseste datele oficiale 2018-2022 si o interpolare catre 2030 bazata pe proiectia 2050 pentru a pastra axa OX pana la 2030.",
        "- Slide-ul 3 foloseste anii primei aprobari majore pentru medicamentele EGFR; pentru Sunvozertinib am folosit 2023 ca primul reper major, chiar daca aprobarea FDA este din 2025.",
        "- Slide-ul 3 foloseste doua imagini separate: un tabel cronologic al peisajului EGFR si un trend separat al publicatiilor PubMed.",
        "- Slide-ul 8 este tabel informativ de studii si metrici, fara comentarii narative.",
        "- Slide-ul 10 include toate studiile din caietul tehnic (reference_library.csv) plus sursele contextuale pentru slide-urile 2-3.",
        "",
        "## PubMed Query",
        PUBMED_MOLECULE_GENERATION_QUERY,
        "",
        "## External Source Links",
        *SOURCE_LINKS,
    ]
    (OUTPUT_DIR / "slide_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for png_path in OUTPUT_DIR.glob("*.png"):
        png_path.unlink()

    metrics = _load_json(REPORTS_DIR / "technical_notebook" / "technical_notebook_metrics.json")
    batch_summary = _load_json(REPORTS_DIR / "prospective_validation_batch.summary.json")
    model_summary = _load_json(REPORTS_DIR / "model_performance_summary.json")
    family_benchmark = _load_json(REPORTS_DIR / "model_family_benchmark.json")
    rediscovery = _load_json(REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_summary.json")
    build_title_template()
    build_problem_chart()
    build_market_table_chart()
    build_pubmed_trend_chart()
    build_chemical_space_chart(metrics, batch_summary)
    build_methodology_chart(model_summary)
    build_model_rmse_chart(model_summary, family_benchmark)
    build_model_r2_chart(model_summary, family_benchmark)
    build_multi_agent_chart(metrics, rediscovery)
    build_ai_vs_studies_chart()
    build_references_chart()
    build_notes_file()

    print(f"[OK] Saved Da Vinci competition graphics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
