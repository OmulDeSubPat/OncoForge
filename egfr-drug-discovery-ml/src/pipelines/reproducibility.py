from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class ArtifactTemplate:
    path: str
    header: tuple[str, ...]
    description: str = ""


STANDARD_METRIC_FILES: tuple[ArtifactTemplate, ...] = (
    ArtifactTemplate("valori_R2.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "R2", "observatii")),
    ArtifactTemplate("valori_RMSE.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "RMSE", "observatii")),
    ArtifactTemplate("valori_MAE.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "MAE", "observatii")),
    ArtifactTemplate("valori_MSE.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "MSE", "observatii")),
    ArtifactTemplate("valori_pIC50.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "pIC50", "observatii")),
    ArtifactTemplate("valori_IC50.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "IC50", "observatii")),
    ArtifactTemplate("valori_Pearson.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "Pearson", "observatii")),
    ArtifactTemplate("valori_Spearman.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "Spearman", "observatii")),
    ArtifactTemplate("valori_Incertitudine.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "Incertitudine", "observatii")),
    ArtifactTemplate("istoric_metrici.csv", ("data", "versiune_model", "nume_experiment", "set_date", "split", "metric_name", "metric_value", "observatii")),
)

STANDARD_BENCHMARK_FILES: tuple[ArtifactTemplate, ...] = (
    ArtifactTemplate("benchmark_studii.csv", ("metoda", "tip_model", "tinta_biologica", "metrica", "valoare", "set_date", "observatii")),
    ArtifactTemplate("comparatii_literatura.csv", ("metoda", "tip_model", "tinta_biologica", "metrica", "valoare", "set_date", "observatii")),
)

STANDARD_TEXT_FILES: dict[str, str] = {
    "legenda_grafice.md": """# Legenda grafice\n\nAcest fisier explica pe scurt ce reprezinta graficele si metricile folosite in proiect. Daca fisierul exista deja, el poate fi extins cu detalii suplimentare pentru prezentare.\n\n## Utilizare\n- Valorile mai mari sunt mai bune doar pentru metricile in care acest lucru este explicit.\n- Pentru eroare, risc sau incertitudine, valori mai mici sunt preferabile.\n- Orice grafic important ar trebui insotit de o scurta interpretare in romana.\n""",
}

METRIC_TEMPLATE_BY_NAME: dict[str, ArtifactTemplate] = {
    template.path.removeprefix("valori_").removesuffix(".csv").lower(): template
    for template in STANDARD_METRIC_FILES
    if template.path.startswith("valori_")
}

ISTORIC_METRICI_HEADER = next(
    template.header for template in STANDARD_METRIC_FILES if template.path == "istoric_metrici.csv"
)


def _ensure_csv(path: Path, header: Iterable[str]) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
    return True


def _ensure_text(path: Path, content: str) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def ensure_standard_reproducibility_files(root: Path | None = None) -> list[Path]:
    base = root or PROJECT_ROOT
    created: list[Path] = []
    for template in (*STANDARD_METRIC_FILES, *STANDARD_BENCHMARK_FILES):
        target = base / template.path
        if _ensure_csv(target, template.header):
            created.append(target)
    for relative_path, content in STANDARD_TEXT_FILES.items():
        target = base / relative_path
        if _ensure_text(target, content):
            created.append(target)
    return created


def append_metric_history(
    *,
    metric_name: str,
    metric_value: float,
    version: str,
    experiment_name: str,
    split: str,
    set_date: str | None = None,
    observations: str = "",
    root: Path | None = None,
) -> list[Path]:
    base = root or PROJECT_ROOT
    ensure_standard_reproducibility_files(base)

    today = set_date or date.today().isoformat()
    metric_key = metric_name.strip()
    if not metric_key:
        raise ValueError("metric_name must not be empty")

    template = METRIC_TEMPLATE_BY_NAME.get(metric_key.lower())
    if template is None:
        raise ValueError(f"Unsupported metric_name: {metric_name!r}")

    metric_row = {
        "data": today,
        "versiune_model": version,
        "nume_experiment": experiment_name,
        "set_date": today,
        "split": split,
        metric_key: metric_value,
        "observatii": observations,
    }

    written: list[Path] = []
    target = base / template.path
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template.header))
        writer.writerow(metric_row)
    written.append(target)

    history_path = base / "istoric_metrici.csv"
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISTORIC_METRICI_HEADER))
        writer.writerow(
            {
                "data": today,
                "versiune_model": version,
                "nume_experiment": experiment_name,
                "set_date": today,
                "split": split,
                "metric_name": metric_key,
                "metric_value": metric_value,
                "observatii": observations,
            }
        )
    written.append(history_path)
    return written


def write_metric_snapshot(
    *,
    metrics: dict[str, float],
    version: str,
    experiment_name: str,
    split: str,
    set_date: str | None = None,
    observations: str = "",
    root: Path | None = None,
) -> list[Path]:
    written: list[Path] = []
    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            continue
        written.extend(
            append_metric_history(
                metric_name=metric_name,
                metric_value=float(metric_value),
                version=version,
                experiment_name=experiment_name,
                split=split,
                set_date=set_date,
                observations=observations,
                root=root,
            )
        )
    return written


def export_reproducibility_manifest(root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    ensure_standard_reproducibility_files(base)
    files = [
        *(base / template.path for template in STANDARD_METRIC_FILES),
        *(base / template.path for template in STANDARD_BENCHMARK_FILES),
        *(base / relative_path for relative_path in STANDARD_TEXT_FILES),
    ]
    manifest = {
        "root": str(base),
        "files": [
            {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
            for path in files
        ],
    }
    return manifest


def write_manifest(path: Path | None = None, root: Path | None = None) -> Path:
    base = root or PROJECT_ROOT
    out_path = path or (base / "reproducibility_manifest.json")
    manifest = export_reproducibility_manifest(base)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path
