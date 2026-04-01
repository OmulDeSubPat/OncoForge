from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from src.config import PROJECT_ROOT

LITERATURE_SOURCE_URLS = [
    "https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00071g",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8796309/",
    "https://pubs.rsc.org/en/content/articlepdf/2019/sc/c8sc05611k",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/",
]


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    if raw is None:
        return float(default)
    try:
        if pd.isna(raw):
            return float(default)
    except TypeError:
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class RoutePenalty:
    label: str
    material_multiplier: float
    step_bonus: float
    labor_bonus: float


ROUTE_PENALTIES = [
    (("late_stage", "tuning", "anisole", "alkoxy", "demethylation"), RoutePenalty("late_stage", 1.00, 0.00, 0.00)),
    (("matched_molecular_pair", "mmp", "bioisostere"), RoutePenalty("mmp", 1.02, 0.08, 0.05)),
    (("snar",), RoutePenalty("snar", 1.08, 0.18, 0.12)),
    (("scaffold_decoration", "decoration"), RoutePenalty("decorare", 1.10, 0.20, 0.14)),
    (("fragment", "growing"), RoutePenalty("fragment_growing", 1.14, 0.28, 0.18)),
    (("linker",), RoutePenalty("linker", 1.16, 0.32, 0.20)),
    (("coupling", "suzuki", "buchwald", "amide"), RoutePenalty("cuplare", 1.18, 0.35, 0.22)),
    (("spiro", "macro", "stereo"), RoutePenalty("complex", 1.24, 0.45, 0.30)),
]


def _route_penalty(row: Mapping[str, Any]) -> RoutePenalty:
    fields = [
        str(row.get("synthetic_route", "") or ""),
        str(row.get("reaction_family", "") or ""),
        str(row.get("action_category", "") or ""),
        str(row.get("action_name", "") or ""),
    ]
    haystack = " ".join(fields).lower()
    for keywords, penalty in ROUTE_PENALTIES:
        if any(keyword in haystack for keyword in keywords):
            return penalty
    return RoutePenalty("standard", 1.05, 0.15, 0.10)


def _compute_structure_defaults(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mw": 400.0,
            "logp": 3.5,
            "ring_count": 3.0,
            "rotatable_bonds": 5.0,
            "stereo_centers": 0.0,
        }
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "stereo_centers": float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    }


def _cost_band(cost_for_10mg: float) -> str:
    if cost_for_10mg < 25:
        return "foarte scazut"
    if cost_for_10mg < 60:
        return "scazut"
    if cost_for_10mg < 150:
        return "mediu"
    if cost_for_10mg < 400:
        return "ridicat"
    return "foarte ridicat"


def estimate_molecule_cost(row: Mapping[str, Any]) -> dict[str, Any]:
    smiles = str(row.get("smiles", "") or "")
    defaults = _compute_structure_defaults(smiles)

    sa_score = _value(row, "SA_score", 4.0)
    mw = _value(row, "MW", defaults["mw"])
    logp = _value(row, "LogP", defaults["logp"])
    ring_count = _value(row, "ring_count", defaults["ring_count"])
    rotatable_bonds = _value(row, "rotatable_bonds", defaults["rotatable_bonds"])
    stereo_centers = _value(row, "stereo_centers", defaults["stereo_centers"])

    top5_train_similarity = _value(row, "top5_train_similarity", 0.45)
    max_train_similarity = _value(row, "max_train_similarity", 0.55)
    novelty_score = _value(row, "novelty_score", max(0.0, 1.0 - max_train_similarity))
    rarity_index = _clamp(0.65 * (1.0 - top5_train_similarity) + 0.35 * novelty_score, 0.0, 1.0)

    synthetic_support = _value(
        row,
        "route_synthetic_support_score",
        _value(row, "synthetic_feasibility_score", 0.60),
    )
    medchem_realism = _value(row, "medchem_realism_score", 0.60)
    transformation_confidence = _value(row, "transformation_confidence_score", 0.60)
    alert_count = _value(row, "alert_count", _value(row, "structural_alert_count", 0.0))
    severe_alert_count = _value(row, "severe_alert_count", 0.0)

    route_penalty = _route_penalty(row)
    complexity_index = _clamp((sa_score - 1.5) / 6.5, 0.0, 1.0)
    stereo_index = _clamp(stereo_centers / 4.0, 0.0, 1.0)
    ring_index = _clamp(max(0.0, ring_count - 2.0) / 4.0, 0.0, 1.0)
    flexibility_index = _clamp(max(0.0, rotatable_bonds - 6.0) / 8.0, 0.0, 1.0)
    logp_index = _clamp(max(0.0, logp - 3.0) / 3.0, 0.0, 1.0)
    mass_index = _clamp(max(0.0, mw - 350.0) / 250.0, 0.0, 1.5)

    purification_complexity = _clamp(
        0.38 * logp_index
        + 0.18 * stereo_index
        + 0.16 * _clamp(alert_count / 4.0, 0.0, 1.0)
        + 0.16 * _clamp(severe_alert_count / 2.0, 0.0, 1.0)
        + 0.12 * flexibility_index,
        0.0,
        1.25,
    )

    estimated_step_count = _clamp(
        1.15
        + 2.30 * complexity_index
        + 0.75 * stereo_index
        + 0.45 * ring_index
        + 0.65 * route_penalty.step_bonus
        + 0.30 * _clamp(severe_alert_count, 0.0, 2.0)
        + 0.20 * purification_complexity,
        1.0,
        8.0,
    )

    estimated_step_yield = _clamp(
        0.85
        - 0.12 * complexity_index
        - 0.05 * stereo_index
        - 0.05 * route_penalty.step_bonus
        + 0.07 * synthetic_support
        + 0.05 * medchem_realism
        + 0.03 * transformation_confidence
        - 0.03 * _clamp(alert_count / 4.0, 0.0, 1.0)
        - 0.04 * _clamp(severe_alert_count / 2.0, 0.0, 1.0),
        0.38,
        0.92,
    )

    effective_steps = max(0, int(round(estimated_step_count)) - 1)
    estimated_yield_penalty = 1.0 / max(estimated_step_yield**effective_steps, 0.05)

    estimated_labor_hours = estimated_step_count * (
        0.85
        + 0.45 * purification_complexity
        + 0.22 * stereo_index
        + 0.20 * route_penalty.labor_bonus
    )
    hourly_rate_usd = 45.0
    operation_cost_usd_per_mmol = estimated_labor_hours * hourly_rate_usd

    material_cost_usd_per_mmol = (
        38.0
        * estimated_step_count
        * (1.0 + 0.30 * mass_index + 0.10 * ring_index)
        * (1.0 + 0.85 * rarity_index)
        * route_penalty.material_multiplier
    )

    purification_multiplier = 1.0 + 0.55 * purification_complexity + 0.12 * _clamp(severe_alert_count, 0.0, 2.0)
    estimated_cost_usd_per_mmol = (operation_cost_usd_per_mmol + material_cost_usd_per_mmol) * estimated_yield_penalty * purification_multiplier
    estimated_cost_score = 1.0 / (1.0 + (estimated_cost_usd_per_mmol / 750.0))
    estimated_cost_for_10mg_usd = estimated_cost_usd_per_mmol * (10.0 / max(mw, 1.0))
    estimated_cost_for_100mg_usd = estimated_cost_usd_per_mmol * (100.0 / max(mw, 1.0))

    return {
        "estimated_route_label": route_penalty.label,
        "estimated_route_multiplier": route_penalty.material_multiplier,
        "estimated_step_count": estimated_step_count,
        "estimated_step_yield": estimated_step_yield,
        "estimated_yield_penalty": estimated_yield_penalty,
        "estimated_rarity_index": rarity_index,
        "estimated_purification_complexity": purification_complexity,
        "estimated_labor_hours": estimated_labor_hours,
        "estimated_operation_cost_usd_per_mmol": operation_cost_usd_per_mmol,
        "estimated_material_cost_usd_per_mmol": material_cost_usd_per_mmol,
        "estimated_purification_multiplier": purification_multiplier,
        "estimated_cost_usd_per_mmol": estimated_cost_usd_per_mmol,
        "estimated_cost_for_10mg_usd": estimated_cost_for_10mg_usd,
        "estimated_cost_for_100mg_usd": estimated_cost_for_100mg_usd,
        "estimated_cost_score": estimated_cost_score,
        "estimated_cost_band": _cost_band(estimated_cost_for_10mg_usd),
    }


def add_cost_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = [estimate_molecule_cost(row) for row in frame.to_dict(orient="records")]
    enriched = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return enriched


def build_cost_model_markdown() -> str:
    return "\n".join(
        [
            "# Formula estimator cost",
            "",
            "Acest estimator este euristic si a fost construit pentru prioritizare, nu pentru cotatie comerciala exacta.",
            "",
            "Baza din literatura:",
            "- Badowski et al. descriu costul unei rute ca suma dintre costurile fixe pe reactie si costurile precursorilor propagate prin randament.",
            "- RouteScore adauga explicit trei axe: timp de lucru, cost monetar si masa materialelor consumate.",
            "- CoPriNet arata ca scorurile SA coreleaza slab cu pretul si nu trebuie folosite singure.",
            "- SAscore ramane util ca proxy pentru dificultatea sintetica si este inclus doar ca una dintre componente.",
            "",
            "Formula practica folosita aici:",
            "1. Estimam numarul de pasi din SA, stereochimie, complexitatea ciclurilor si tipul transformarii.",
            "2. Estimam randamentul per pas din suportul sintetic, realismul med-chem si increderea transformarii.",
            "3. Estimam costul operational din ore de lucru si costul material din masa moleculara, raritatea structurala si ruta.",
            "4. Corectam costul total cu o penalizare de randament si o penalizare de purificare.",
            "",
            "Surse:",
            *[f"- {url}" for url in LITERATURE_SOURCE_URLS],
            "",
            f"Fisier generat din repo: {Path(PROJECT_ROOT).name}",
        ]
    )
