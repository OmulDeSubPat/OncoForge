from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.config import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "reports" / "presentation_key_visuals"


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
        linewidth=2.2,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=13)


def build_impact_line() -> None:
    years = np.array([2012, 2018, 2020, 2022])
    lung_cases = np.array([1820000, 2094000, 2206771, 2480675], dtype=float)
    egfr_prev = 0.493
    low = lung_cases * 0.10 * egfr_prev
    mid = lung_cases * 0.15 * egfr_prev
    high = lung_cases * 0.20 * egfr_prev

    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.fill_between(years, low, high, color="#BFDBFE", alpha=0.75, linewidth=0)
    ax.plot(years, mid, color="#2563EB", linewidth=4, marker="o", markersize=10)
    ax.scatter(years, mid, color="#1D4ED8", s=90, zorder=3)

    for year, value in zip(years, mid):
        ax.text(year, value + 9000, f"{int(round(value)):,}".replace(",", "."), ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_title("Cazuri asociate EGFR per an", pad=14)
    ax.set_ylabel("Numar estimat de cazuri noi")
    ax.set_xticks(years)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "01_impact_egfr_line.png")


def build_impact_bars() -> None:
    years = ["2012", "2018", "2020", "2022"]
    lung_cases = np.array([1820000, 2094000, 2206771, 2480675], dtype=float)
    egfr_prev = 0.493
    estimates = lung_cases * 0.15 * egfr_prev
    colors = ["#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    bars = ax.bar(years, estimates, color=colors, width=0.60)
    ax.set_title("Cazuri asociate EGFR per an", pad=14)
    ax.set_ylabel("Numar estimat de cazuri noi")

    for bar, value in zip(bars, estimates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(estimates) * 0.025,
            f"{int(round(value)):,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "02_impact_egfr_bars.png")


def build_pipeline_horizontal() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.6))
    ax.axis("off")
    fig.suptitle("Pipeline-ul Sistemului", fontsize=28, fontweight="bold", y=0.96)

    steps = [
        (0.02, "#DBEAFE", "1. Colectare\ndate"),
        (0.22, "#DCFCE7", "2. Curatare\nsi filtrare"),
        (0.42, "#FEF3C7", "3. Antrenare\nmodel"),
        (0.62, "#FCE7F3", "4. Generare\nmolecule"),
        (0.82, "#E9D5FF", "5. Ranking si\nselectie"),
    ]

    y = 0.22
    w = 0.15
    h = 0.42

    for i, (x, color, text) in enumerate(steps):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=2.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=17, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 0.01, y + h / 2),
                xytext=(x + w, y + h / 2),
                arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"},
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "03_pipeline_horizontal.png")


def build_pipeline_detailed() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.axis("off")
    fig.suptitle("Cum Functioneaza Sistemul", fontsize=28, fontweight="bold", y=0.96)

    _card(ax, 0.06, 0.66, 0.25, 0.18, "#DBEAFE", "Date publice", "ChEMBL, BindingDB,\nPapyrus, ExCAPE")
    _card(ax, 0.37, 0.66, 0.25, 0.18, "#DCFCE7", "Prelucrare", "curatare, standardizare,\nfiltrare")
    _card(ax, 0.68, 0.66, 0.25, 0.18, "#FEF3C7", "Model AI", "invata relatia dintre\nstructura si activitate")

    _card(ax, 0.22, 0.28, 0.25, 0.18, "#FCE7F3", "Generare", "propune molecule\nnoi")
    _card(ax, 0.53, 0.28, 0.25, 0.18, "#E9D5FF", "Selectie finala", "pastreaza doar\ncele mai bune variante")

    ax.annotate("", xy=(0.37, 0.75), xytext=(0.31, 0.75), arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"})
    ax.annotate("", xy=(0.68, 0.75), xytext=(0.62, 0.75), arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"})
    ax.annotate("", xy=(0.35, 0.46), xytext=(0.73, 0.66), arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"})
    ax.annotate("", xy=(0.66, 0.46), xytext=(0.47, 0.37), arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#475569"})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "04_pipeline_detailed.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    build_impact_line()
    build_impact_bars()
    build_pipeline_horizontal()
    build_pipeline_detailed()
    print(f"[OK] Saved key visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
