from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

from src.config import PROJECT_ROOT
from src.economics.cost_model import estimate_molecule_cost
from src.gui.data_access import prepare_molecule_frame, safe_csv, safe_float, safe_json, tail_text
from src.gui.live_generation_worker import MODE_LABELS
from src.gui.oncoforge_api.runtime import is_worker_pid_active, resolve_session_dir

MARKET_PRIORITY = ["Osimertinib", "Gefitinib", "Erlotinib", "Afatinib", "Dacomitinib", "Lazertinib", "Sunvozertinib"]


def _status_label(raw_status: str) -> str:
    mapping = {
        "pornit": "Pornit",
        "in_rulare": "Activ",
        "finalizat": "Finalizat",
        "oprit": "Oprit",
        "eroare": "Eroare",
    }
    return mapping.get(str(raw_status or "").lower(), "Pregatire")


def _is_running_status(raw_status: str) -> bool:
    return str(raw_status or "").lower() in {"pornit", "in_rulare"}


def _normalize_status_payload(status: dict[str, Any]) -> dict[str, Any]:
    if not status:
        return {}
    normalized = dict(status)
    raw_status = str(normalized.get("status", "")).lower()
    pid = int(normalized.get("pid", 0) or 0)
    if raw_status in {"pornit", "in_rulare"} and not is_worker_pid_active(pid):
        normalized["status"] = "oprit"
        normalized["mesaj"] = "Procesul worker nu mai este activ. Sesiunea poate fi pornita din nou."
    return normalized


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, "Sesiune OncoSynth")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "da", "yes"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text or default


def _read_frames(session_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    status = _normalize_status_payload(safe_json(session_dir / "status.json"))
    molecules_df = prepare_molecule_frame(safe_csv(session_dir / "molecule_generate.csv"))
    rounds_df = safe_csv(session_dir / "rezumat_runde.csv")
    return status, molecules_df, rounds_df


def _best_row(molecules_df: pd.DataFrame) -> pd.Series | None:
    if molecules_df.empty:
        return None
    ranked = molecules_df.sort_values(["live_rank_score", "predicted_pIC50", "QED"], ascending=[False, False, False])
    return None if ranked.empty else ranked.iloc[0]


def _select_row(molecules_df: pd.DataFrame, rank: int | None, smiles: str | None) -> pd.Series | None:
    if molecules_df.empty:
        return None
    if smiles:
        matched = molecules_df[molecules_df["smiles"].fillna("").astype(str) == smiles]
        if not matched.empty:
            return matched.iloc[0]
    if rank is not None and "rank" in molecules_df.columns:
        matched = molecules_df[pd.to_numeric(molecules_df["rank"], errors="coerce") == rank]
        if not matched.empty:
            return matched.iloc[0]
    return _best_row(molecules_df)


def _build_summary(status: dict[str, Any], molecules_df: pd.DataFrame) -> dict[str, Any]:
    if molecules_df.empty:
        return {
            "moleculeCount": 0,
            "promotedCount": 0,
            "reviewCount": 0,
            "rejectedCount": 0,
            "bestPic50": 0.0,
            "bestScore": safe_float(status.get("best_score"), 0.0),
            "meanQed": 0.0,
        }

    live_status = molecules_df.get("live_status", pd.Series(dtype=str)).fillna("necunoscut")
    counts = live_status.value_counts().to_dict()
    return {
        "moleculeCount": int(len(molecules_df)),
        "promotedCount": int(counts.get("promovata", 0)),
        "reviewCount": int(counts.get("revizie", 0)),
        "rejectedCount": int(counts.get("respinsa", 0)),
        "bestPic50": round(safe_float(molecules_df["predicted_pIC50"].max(), 0.0), 3),
        "bestScore": round(safe_float(molecules_df["live_rank_score"].max(), safe_float(status.get("best_score"), 0.0)), 3),
        "meanQed": round(safe_float(molecules_df["QED"].mean(), 0.0), 3),
    }


def _agent_component_scores(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {"generator": 0.0, "toxicity": 0.0, "validator": 0.0, "optimizer": 0.0}

    generator = _clamp(safe_float(row.get("generator_priority_score"), 0.0))
    toxicity = _clamp(
        (1.0 - safe_float(row.get("reward_hacking_risk"), 0.0)) * 0.7
        + (1.0 - min(1.0, safe_float(row.get("structural_alert_count"), 0.0) / 3.0)) * 0.3
    )
    validator = _clamp(
        safe_float(row.get("QED"), 0.0) * 0.35
        + safe_float(row.get("synthetic_feasibility_score"), 0.0) * 0.35
        + (0.3 if _boolish(row.get("audit_pass")) else 0.08)
    )
    optimizer = _clamp(safe_float(row.get("predicted_pIC50"), 0.0) / 10.0)
    return {
        "generator": generator,
        "toxicity": toxicity,
        "validator": validator,
        "optimizer": optimizer,
    }


def _agent_contributions(row: pd.Series | None) -> list[dict[str, Any]]:
    if row is None:
        return [
            {"id": "generator", "name": "Molecular Generator", "status": "idle", "contribution": 0.0, "headline": "Asteapta prima generatie.", "lastAction": "Fara evenimente inca"},
            {"id": "toxicity", "name": "Toxicity Evaluator", "status": "idle", "contribution": 0.0, "headline": "Asteapta prima evaluare.", "lastAction": "Fara penalizari calculate"},
            {"id": "validator", "name": "Chemical Validator", "status": "idle", "contribution": 0.0, "headline": "Asteapta prima verificare.", "lastAction": "Fara audit chimic"},
            {"id": "optimizer", "name": "Potency Optimizer", "status": "idle", "contribution": 0.0, "headline": "Asteapta primul scor pIC50.", "lastAction": "Niciun candidat optimizat"},
        ]

    contributions = _agent_component_scores(row)
    total = sum(contributions.values()) or 1.0
    return [
        {
            "id": "generator",
            "name": "Molecular Generator",
            "status": "active",
            "contribution": round(contributions["generator"] / total, 4),
            "headline": f"Transformare selectata: {_safe_str(row.get('action_name'), 'necunoscuta')}",
            "lastAction": _safe_str(row.get("synthetic_route"), "generare noua"),
        },
        {
            "id": "toxicity",
            "name": "Toxicity Evaluator",
            "status": "monitoring",
            "contribution": round(contributions["toxicity"] / total, 4),
            "headline": f"Risc verificat: {safe_float(row.get('reward_hacking_risk'), 0.0):.2f}",
            "lastAction": _safe_str(row.get("reward_hacking_flags"), "fara alerte"),
        },
        {
            "id": "validator",
            "name": "Chemical Validator",
            "status": "active",
            "contribution": round(contributions["validator"] / total, 4),
            "headline": f"Audit: {_safe_str(row.get('audit_status'), 'pass')}",
            "lastAction": _safe_str(row.get("hard_constraint_notes"), "Constrangeri respectate"),
        },
        {
            "id": "optimizer",
            "name": "Potency Optimizer",
            "status": "training",
            "contribution": round(contributions["optimizer"] / total, 4),
            "headline": f"pIC50 prezis: {safe_float(row.get('predicted_pIC50'), 0.0):.2f}",
            "lastAction": f"Delta vs parinte: {safe_float(row.get('delta_predicted_pIC50'), 0.0):+.2f}",
        },
    ]


def _agent_flows(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = {agent["id"]: float(agent["contribution"]) for agent in agents}
    return [
        {"source": "generator", "target": "validator", "weight": round((weights.get("generator", 0.0) + weights.get("validator", 0.0)) / 2, 3)},
        {"source": "validator", "target": "toxicity", "weight": round((weights.get("validator", 0.0) + weights.get("toxicity", 0.0)) / 2, 3)},
        {"source": "toxicity", "target": "optimizer", "weight": round((weights.get("toxicity", 0.0) + weights.get("optimizer", 0.0)) / 2, 3)},
        {"source": "optimizer", "target": "generator", "weight": round((weights.get("optimizer", 0.0) + weights.get("generator", 0.0)) / 2, 3)},
    ]


def _build_overview(status: dict[str, Any], molecules_df: pd.DataFrame, rounds_df: pd.DataFrame) -> dict[str, Any]:
    best_row = _best_row(molecules_df)
    summary = _build_summary(status, molecules_df)

    total_rounds = max(1, int(status.get("total_rounds", 1) or 1))
    total_seeds = max(1, int(status.get("total_seeds", 1) or 1))
    current_round = int(status.get("current_round", 0) or 0)
    current_seed = int(status.get("current_seed", 0) or 0)
    progress = min(1.0, ((max(0, current_round - 1) * total_seeds) + current_seed) / (total_rounds * total_seeds))

    best_payload = None
    if best_row is not None:
        best_payload = {
            "smiles": _safe_str(best_row.get("smiles")),
            "status": _safe_str(best_row.get("live_status")),
            "action": _safe_str(best_row.get("action_name")),
            "score": round(safe_float(best_row.get("live_rank_score"), 0.0), 3),
            "pic50": round(safe_float(best_row.get("predicted_pIC50"), 0.0), 3),
            "uncertainty": round(safe_float(best_row.get("uncertainty"), 0.0), 3),
            "qed": round(safe_float(best_row.get("QED"), 0.0), 3),
            "syntheticFeasibility": round(safe_float(best_row.get("synthetic_feasibility_score"), 0.0), 3),
            "marketReference": _safe_str(best_row.get("closest_market_name"), "necunoscut"),
            "marketSimilarity": round(safe_float(best_row.get("max_market_similarity"), 0.0), 3),
            "cost10mg": round(safe_float(best_row.get("estimated_cost_for_10mg_usd"), 0.0), 2),
        }

    latest_round = None
    if not rounds_df.empty:
        latest = rounds_df.iloc[-1]
        latest_round = {
            "round": int(safe_float(latest.get("runda"), 0)),
            "seedStep": int(safe_float(latest.get("pas_seed"), 0)),
            "newCandidates": int(safe_float(latest.get("candidati_noi"), 0)),
            "promotedCandidates": int(safe_float(latest.get("candidati_promovati"), 0)),
            "bestScore": round(safe_float(latest.get("scor_live_maxim"), 0.0), 3),
        }

    session_name = Path(status.get("session_dir", "")).name if status.get("session_dir") else resolve_session_dir().name
    return {
        "sessionName": session_name,
        "mode": _safe_str(status.get("mod"), "ghidat_ai"),
        "modeLabel": _safe_str(status.get("mod_label"), _mode_label(_safe_str(status.get("mod"), "ghidat_ai"))),
        "status": _safe_str(status.get("status"), "pregatire"),
        "statusLabel": _status_label(_safe_str(status.get("status"), "pregatire")),
        "message": _safe_str(status.get("mesaj"), "Platforma este pregatita pentru o sesiune noua."),
        "updatedAt": _safe_str(status.get("updated_at")),
        "running": _is_running_status(_safe_str(status.get("status"))),
        "progress": round(progress, 4),
        "summary": summary,
        "bestMolecule": best_payload,
        "latestRound": latest_round,
    }


def _build_metrics(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {"primary": [], "radar": [], "riskFlags": [], "comparison": None}

    primary = [
        {"label": "pIC50", "value": round(safe_float(row.get("predicted_pIC50"), 0.0), 3), "tone": "primary"},
        {"label": "Incertitudine", "value": round(safe_float(row.get("uncertainty"), 0.0), 3), "tone": "warning"},
        {"label": "QED", "value": round(safe_float(row.get("QED"), 0.0), 3), "tone": "success"},
        {"label": "SA score", "value": round(safe_float(row.get("SA_score"), 0.0), 3), "tone": "info"},
        {"label": "MW", "value": round(safe_float(row.get("MW"), 0.0), 2), "tone": "neutral"},
        {"label": "LogP", "value": round(safe_float(row.get("LogP"), 0.0), 2), "tone": "neutral"},
        {"label": "TPSA", "value": round(safe_float(row.get("TPSA"), 0.0), 2), "tone": "neutral"},
        {"label": "HBD/HBA", "value": f"{int(safe_float(row.get('HBD'), 0))}/{int(safe_float(row.get('HBA'), 0))}", "tone": "neutral"},
    ]
    radar = [
        {"axis": "Potenta", "value": round(_clamp(safe_float(row.get("predicted_pIC50"), 0.0) / 10.0), 3)},
        {"axis": "Validitate", "value": round(_clamp(safe_float(row.get("QED"), 0.0)), 3)},
        {"axis": "Sinteza", "value": round(_clamp(safe_float(row.get("synthetic_feasibility_score"), 0.0)), 3)},
        {"axis": "Noutate", "value": round(_clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0)), 3)},
        {"axis": "Certitudine", "value": round(_clamp(1.0 - safe_float(row.get("uncertainty"), 0.0)), 3)},
        {"axis": "Siguranta", "value": round(_clamp(1.0 - safe_float(row.get("reward_hacking_risk"), 0.0)), 3)},
    ]

    flags: list[dict[str, str]] = []
    if _boolish(row.get("has_PAINS")):
        flags.append({"label": "PAINS", "tone": "danger"})
    if safe_float(row.get("structural_alert_count"), 0.0) > 0:
        flags.append({"label": f"Alerte structurale: {int(safe_float(row.get('structural_alert_count'), 0.0))}", "tone": "danger"})
    if safe_float(row.get("reward_hacking_risk"), 0.0) > 0.2:
        flags.append({"label": "Necesita verificare de reward hacking", "tone": "warning"})
    if not flags:
        flags.append({"label": "Fara alerte majore", "tone": "success"})

    comparison = {
        "referenceName": _safe_str(row.get("closest_market_name"), "Comparator de piata"),
        "similarity": round(safe_float(row.get("max_market_similarity"), 0.0), 3),
        "novelty": round(_clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0)), 3),
        "marketSupport": round(safe_float(row.get("market_novelty_score"), 0.0), 3),
    }
    return {"primary": primary, "radar": radar, "riskFlags": flags, "comparison": comparison}


@lru_cache(maxsize=1)
def _load_marketed_reference_frame() -> pd.DataFrame:
    reports_dir = PROJECT_ROOT / "reports"
    frame = safe_csv(reports_dir / "marketed_egfr_scored.csv")
    if frame.empty:
        frame = safe_csv(reports_dir / "marketed_egfr_structural_benchmark.csv")
    if frame.empty:
        return frame

    out = frame.copy()
    if "estimated_cost_for_10mg_usd" not in out.columns:
        estimates = [estimate_molecule_cost(row) for row in out.to_dict(orient="records")]
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(estimates)], axis=1)
    out["display_name"] = out.get("name", pd.Series([""] * len(out))).fillna("")
    out.loc[out["display_name"] == "", "display_name"] = out.get("closest_market_name", pd.Series([""] * len(out))).fillna("")
    out["display_name"] = out["display_name"].replace("", "Comparator piata")
    return out


def _normalized_market_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "potency": round(_clamp(safe_float(row.get("predicted_pIC50"), 0.0) / 10.0), 3),
        "qed": round(_clamp(safe_float(row.get("QED"), 0.0)), 3),
        "sa": round(_clamp(1.0 - max(0.0, safe_float(row.get("SA_score"), 0.0) - 1.0) / 6.0), 3),
        "cost": round(_clamp(1.0 / (1.0 + safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0) / 35.0)), 3),
        "novelty": round(_clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0)), 3),
        "risk": round(_clamp(1.0 - safe_float(row.get("reward_hacking_risk"), 0.0)), 3),
    }


def _market_entry_from_row(row: pd.Series, *, kind: str, label: str | None = None) -> dict[str, Any]:
    name = label or _safe_str(row.get("name"), _safe_str(row.get("closest_market_name"), "Comparator piata"))
    return {
        "id": f"{kind}-{abs(hash(_safe_str(row.get('smiles'), name)))}",
        "name": name,
        "kind": kind,
        "referenceClass": _safe_str(row.get("class"), "egfr_tki"),
        "smiles": _safe_str(row.get("smiles")),
        "raw": {
            "potency": round(safe_float(row.get("predicted_pIC50"), 0.0), 3),
            "qed": round(safe_float(row.get("QED"), 0.0), 3),
            "sa": round(safe_float(row.get("SA_score"), 0.0), 3),
            "cost": round(safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0), 2),
            "novelty": round(_clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0)), 3),
            "risk": round(safe_float(row.get("reward_hacking_risk"), 0.0), 3),
        },
        "normalized": _normalized_market_metrics(row),
    }


def _build_market_compare(row: pd.Series | None) -> dict[str, Any]:
    marketed = _load_marketed_reference_frame()
    entries: list[dict[str, Any]] = []
    if row is not None:
        entries.append(_market_entry_from_row(row, kind="selectata", label="Candidatul selectat"))

    comparators: list[pd.Series] = []
    if not marketed.empty:
        lower_names = marketed["display_name"].fillna("").astype(str).str.lower()
        for market_name in MARKET_PRIORITY[:3]:
            matched = marketed[lower_names == market_name.lower()]
            if not matched.empty:
                comparators.append(matched.iloc[0])
        if len(comparators) < 3:
            existing = {str(item.get("display_name", "")) for item in comparators}
            fallback = marketed.sort_values(["final_score", "predicted_pIC50"], ascending=[False, False])
            for _, candidate in fallback.iterrows():
                name = _safe_str(candidate.get("display_name"))
                if name and name not in existing:
                    comparators.append(candidate)
                    existing.add(name)
                if len(comparators) >= 3:
                    break

    entries.extend(_market_entry_from_row(candidate, kind="comparator") for candidate in comparators)
    return {
        "candidateSmiles": _safe_str(row.get("smiles")) if row is not None else "",
        "axes": ["Potenta", "QED", "SA", "Cost", "Noutate", "Risc"],
        "entries": entries,
    }


def _build_thresholds(row: pd.Series) -> list[dict[str, Any]]:
    return [
        {"label": "QED minim", "passed": safe_float(row.get("QED"), 0.0) >= 0.45, "value": f"{safe_float(row.get('QED'), 0.0):.2f}", "reference": ">= 0.45"},
        {"label": "Incertitudine controlata", "passed": safe_float(row.get("uncertainty"), 1.0) <= 0.12, "value": f"{safe_float(row.get('uncertainty'), 0.0):.3f}", "reference": "<= 0.12"},
        {"label": "Fezabilitate sintetica", "passed": safe_float(row.get("synthetic_feasibility_score"), 0.0) >= 0.65, "value": f"{safe_float(row.get('synthetic_feasibility_score'), 0.0):.2f}", "reference": ">= 0.65"},
        {"label": "Fara PAINS", "passed": not _boolish(row.get("has_PAINS")), "value": "da" if not _boolish(row.get("has_PAINS")) else "nu", "reference": "da"},
        {"label": "Alerte structurale", "passed": safe_float(row.get("structural_alert_count"), 0.0) == 0.0, "value": f"{int(safe_float(row.get('structural_alert_count'), 0.0))}", "reference": "0"},
        {"label": "Cost screening 10 mg", "passed": safe_float(row.get("estimated_cost_for_10mg_usd"), 9999.0) <= 60.0, "value": f"${safe_float(row.get('estimated_cost_for_10mg_usd'), 0.0):.2f}", "reference": "<= $60"},
        {"label": "Diferenta fata de piata", "passed": safe_float(row.get("max_market_similarity"), 1.0) <= 0.85, "value": f"{safe_float(row.get('max_market_similarity'), 0.0):.2f}", "reference": "<= 0.85"},
    ]


def _build_explainability(row: pd.Series, agents: list[dict[str, Any]]) -> dict[str, Any]:
    pros: list[str] = []
    cons: list[str] = []
    pic50 = safe_float(row.get("predicted_pIC50"), 0.0)
    qed = safe_float(row.get("QED"), 0.0)
    synth = safe_float(row.get("synthetic_feasibility_score"), 0.0)
    cost = safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0)
    novelty = _clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0))
    risk = safe_float(row.get("reward_hacking_risk"), 0.0)
    delta_score = safe_float(row.get("delta_final_score"), 0.0)
    uncertainty = safe_float(row.get("uncertainty"), 0.0)
    structural_alerts = int(safe_float(row.get("structural_alert_count"), 0.0))

    if pic50 >= 9.0:
        pros.append(f"Potenta prezisa ridicata ({pic50:.2f} pIC50).")
    if synth >= 0.7:
        pros.append(f"Fezabilitate sintetica buna ({synth:.2f}).")
    if qed >= 0.5:
        pros.append(f"Profil molecular bun prin QED ({qed:.2f}).")
    if cost <= 25:
        pros.append(f"Cost de screening accesibil (${cost:.2f} pentru 10 mg).")
    if novelty >= 0.35:
        pros.append(f"Noutate reala fata de comparatorii de piata ({novelty:.2f}).")
    if delta_score > 0:
        pros.append(f"A urcat fata de parinte prin scorul final ({delta_score:+.2f}).")

    if safe_float(row.get("max_market_similarity"), 0.0) >= 0.8:
        cons.append("Este foarte apropiata de molecule deja existente pe piata.")
    if risk >= 0.18:
        cons.append(f"Risc guardrail mai mare decat idealul ({risk:.2f}).")
    if uncertainty >= 0.12:
        cons.append(f"Incertitudinea modelului este ridicata ({uncertainty:.3f}).")
    if cost >= 60:
        cons.append(f"Costul de screening pentru 10 mg este ridicat (${cost:.2f}).")
    if structural_alerts > 0:
        cons.append(f"Exista {structural_alerts} alerta structurala de clarificat.")
    if delta_score < 0:
        cons.append(f"Scorul a scazut fata de parintele direct ({delta_score:+.2f}).")

    penalties = []
    for label, value in [
        ("Penalizare risc", safe_float(row.get("ranking_penalizare_risc"), 0.0)),
        ("Penalty incertitudine", safe_float(row.get("uncertainty_penalty"), 0.0)),
        ("Penalty copiere piata", safe_float(row.get("market_copy_penalty"), 0.0)),
        ("Penalty reactivitate", safe_float(row.get("reactivity_penalty"), 0.0)),
        ("Penalty guardrail", safe_float(row.get("anti_hacking_penalty"), 0.0)),
    ]:
        if abs(value) > 1e-6:
            penalties.append({"label": label, "value": round(value, 3), "tone": "negative"})

    dominant_agent = max(agents, key=lambda entry: float(entry.get("contribution", 0.0)), default={"name": "N/A"})
    summary = (
        "Molecula a fost promovata pentru combinatia dintre potenta, sinteza si scor final."
        if _safe_str(row.get("live_status")) == "promovata"
        else "Molecula ramane interesanta, dar are cateva semnale care cer verificare inainte de shortlist."
    )
    return {
        "pros": (pros or ["Nu exista inca motive foarte puternice de promovare."])[:3],
        "cons": (cons or ["Nu exista blocaje majore; molecula poate fi comparata direct cu alti lead-uri."])[:3],
        "dominantAgent": dominant_agent.get("name", "N/A"),
        "penalties": penalties,
        "thresholds": _build_thresholds(row),
        "summary": summary,
    }


def _build_admet(row: pd.Series) -> dict[str, Any]:
    has_pains = _boolish(row.get("has_PAINS"))
    structural_alerts = int(safe_float(row.get("structural_alert_count"), 0.0))
    logp = safe_float(row.get("LogP"), 0.0)
    mw = safe_float(row.get("MW"), 0.0)
    tpsa = safe_float(row.get("TPSA"), 0.0)
    warheads = int(safe_float(row.get("covalent_warhead_count"), 0.0))
    lipinski = int(safe_float(row.get("lipinski_violations"), 0.0))
    liabilities = [
        {"label": "PAINS", "tone": "danger" if has_pains else "success", "value": "prezent" if has_pains else "absent", "note": "Semnal clasic de interferenta chimica."},
        {"label": "Alerte structurale", "tone": "danger" if structural_alerts > 0 else "success", "value": str(structural_alerts), "note": "Numarul de alerte detectate in guardrails."},
        {"label": "LogP in afara zonei", "tone": "warning" if logp > 4.5 else "success", "value": f"{logp:.2f}", "note": "Valori mari cresc riscul de solubilitate si off-target."},
        {"label": "Masa moleculara", "tone": "warning" if mw > 500 else "success", "value": f"{mw:.1f}", "note": "Peste 500 Da intra in zona mai dificila pentru lead-like."},
        {"label": "TPSA", "tone": "warning" if tpsa > 120 else "success", "value": f"{tpsa:.1f}", "note": "TPSA mare poate afecta permeabilitatea."},
        {"label": "Warhead / reactivitate", "tone": "warning" if warheads > 0 else "success", "value": str(warheads), "note": "Grupari reactive sau covalente de verificat separat."},
        {"label": "Presiune wild-type proxy", "tone": "warning" if safe_float(row.get("max_market_similarity"), 0.0) > 0.75 else "success", "value": f"{safe_float(row.get('max_market_similarity'), 0.0):.2f}", "note": "Proxy euristic bazat pe similaritatea fata de TKI-uri comerciale."},
        {"label": "Lipinski", "tone": "warning" if lipinski > 0 else "success", "value": str(lipinski), "note": "Numarul de abateri fata de regulile uzuale lead-like."},
    ]
    wild_type_proxy = _clamp(0.55 * safe_float(row.get("max_market_similarity"), 0.0) + 0.20 * min(1.0, warheads / 2.0) + 0.25 * _clamp(max(0.0, logp - 3.5) / 2.5))
    reactivity_risk = _clamp(0.5 * min(1.0, warheads / 2.0) + 0.3 * min(1.0, structural_alerts / 3.0) + 0.2 * min(1.0, lipinski / 2.0))
    summary = "Profil liability controlat pentru screening." if max(wild_type_proxy, reactivity_risk) < 0.45 else "Exista cateva liabilities care cer verificare experimentala sau manuala."
    return {
        "summary": summary,
        "wildTypeProxy": round(wild_type_proxy, 3),
        "reactivityRisk": round(reactivity_risk, 3),
        "liabilities": liabilities,
    }


def _build_decision_history(row: pd.Series, molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    lineage = [_safe_str(item) for item in _safe_str(row.get("lineage_path")).split("->") if _safe_str(item)]
    if not lineage:
        lineage = [_safe_str(row.get("smiles"))]

    known_rows = {str(entry.get("smiles")): entry for _, entry in molecules_df.iterrows()}
    for index, lineage_smiles in enumerate(lineage):
        lineage_row = known_rows.get(lineage_smiles)
        if lineage_row is None:
            continue
        round_idx = int(safe_float(lineage_row.get("round"), 0))
        events.append(
            {
                "id": f"lineage-{index}-{abs(hash(lineage_smiles))}",
                "title": "Pas in linia de evolutie",
                "detail": f"Runda {round_idx}: {_safe_str(lineage_row.get('action_name'), 'molecula parinte')} | scor {safe_float(lineage_row.get('live_rank_score'), 0.0):.2f}",
                "timestamp": "",
                "category": "linie",
                "tone": "info" if index < len(lineage) - 1 else "success",
                "round": round_idx,
            }
        )

    events.extend(
        [
            {
                "id": f"audit-{abs(hash(_safe_str(row.get('smiles'))))}",
                "title": "Audit chimic si reward",
                "detail": f"Audit {_safe_str(row.get('audit_status'), 'pass')} | risc {safe_float(row.get('reward_hacking_risk'), 0.0):.2f} | verified reward {safe_float(row.get('verified_reward'), 0.0):.2f}",
                "timestamp": "",
                "category": "audit",
                "tone": "success" if _boolish(row.get("audit_pass")) else "warning",
                "round": int(safe_float(row.get("round"), 0)),
            },
            {
                "id": f"status-{abs(hash(_safe_str(row.get('smiles'))))}",
                "title": "Decizie de triere",
                "detail": f"Status {_safe_str(row.get('live_status'), 'necunoscut')} | rang #{int(safe_float(row.get('rank'), 0))} | delta scor {safe_float(row.get('delta_final_score'), 0.0):+.2f}",
                "timestamp": "",
                "category": "triere",
                "tone": "success" if _safe_str(row.get("live_status")) == "promovata" else "warning",
                "round": int(safe_float(row.get("round"), 0)),
            },
            {
                "id": f"market-{abs(hash(_safe_str(row.get('smiles'))))}",
                "title": "Comparator piata",
                "detail": f"Cea mai apropiata referinta este {_safe_str(row.get('closest_market_name'), 'necunoscuta')} cu similaritate {safe_float(row.get('max_market_similarity'), 0.0):.2f}.",
                "timestamp": "",
                "category": "comparatie",
                "tone": "info",
                "round": int(safe_float(row.get("round"), 0)),
            },
        ]
    )

    if _safe_str(row.get("cross_database_status")):
        events.append(
            {
                "id": f"evidence-{abs(hash(_safe_str(row.get('smiles'))))}",
                "title": "Suport din baze externe",
                "detail": f"Cross-database: {_safe_str(row.get('cross_database_status'))} | surse independente {int(safe_float(row.get('cross_database_independent_support_count'), 0))}.",
                "timestamp": "",
                "category": "evidenta",
                "tone": "success" if safe_float(row.get("cross_database_consensus_score"), 0.0) >= 0.5 else "warning",
                "round": int(safe_float(row.get("round"), 0)),
            }
        )

    return events


def _build_library(molecules_df: pd.DataFrame, limit: int, search: str, status_filter: str) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []

    filtered = molecules_df.copy()
    if status_filter and status_filter != "all":
        filtered = filtered[filtered["live_status"].fillna("") == status_filter]
    if search:
        query = search.lower()
        market_col = filtered.get("closest_market_name", pd.Series([""] * len(filtered)))
        filtered = filtered[
            filtered["smiles"].fillna("").astype(str).str.lower().str.contains(query, na=False)
            | filtered["transformare_afisare"].fillna("").astype(str).str.lower().str.contains(query, na=False)
            | market_col.fillna("").astype(str).str.lower().str.contains(query, na=False)
        ]

    filtered = filtered.sort_values(["live_rank_score", "predicted_pIC50"], ascending=[False, False]).head(limit)
    library: list[dict[str, Any]] = []
    for _, row in filtered.iterrows():
        library.append(
            {
                "id": f"rank-{int(safe_float(row.get('rank'), 0))}-{abs(hash(_safe_str(row.get('smiles'))))}",
                "rank": int(safe_float(row.get("rank"), 0)),
                "smiles": _safe_str(row.get("smiles")),
                "parent": _safe_str(row.get("parent_seed")),
                "round": int(safe_float(row.get("round"), 0)),
                "status": _safe_str(row.get("live_status")),
                "statusLabel": _safe_str(row.get("stare_afisare")),
                "score": round(safe_float(row.get("live_rank_score"), 0.0), 3),
                "pic50": round(safe_float(row.get("predicted_pIC50"), 0.0), 3),
                "qed": round(safe_float(row.get("QED"), 0.0), 3),
                "uncertainty": round(safe_float(row.get("uncertainty"), 0.0), 3),
                "cost10mg": round(safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0), 2),
                "action": _safe_str(row.get("action_name")),
                "route": _safe_str(row.get("synthetic_route")),
                "marketReference": _safe_str(row.get("closest_market_name")),
                "saScore": round(safe_float(row.get("SA_score"), 0.0), 3),
                "syntheticFeasibility": round(safe_float(row.get("synthetic_feasibility_score"), 0.0), 3),
                "marketSimilarity": round(safe_float(row.get("max_market_similarity"), 0.0), 3),
                "novelty": round(_clamp(1.0 - safe_float(row.get("max_market_similarity"), 0.0)), 3),
                "risk": round(safe_float(row.get("reward_hacking_risk"), 0.0), 3),
                "pains": _boolish(row.get("has_PAINS")),
                "structuralAlerts": int(safe_float(row.get("structural_alert_count"), 0.0)),
                "verifiedReward": round(safe_float(row.get("verified_reward"), 0.0), 3),
                "mw": round(safe_float(row.get("MW"), 0.0), 2),
                "logP": round(safe_float(row.get("LogP"), 0.0), 2),
                "tpsa": round(safe_float(row.get("TPSA"), 0.0), 2),
                "auditPass": _boolish(row.get("audit_pass")),
                "generatorPriority": round(safe_float(row.get("generator_priority_score"), 0.0), 3),
            }
        )
    return library


def _build_timeline(molecules_df: pd.DataFrame, rounds_df: pd.DataFrame) -> dict[str, Any]:
    generations: list[dict[str, Any]] = []
    for _, row in rounds_df.iterrows():
        generations.append(
            {
                "round": int(safe_float(row.get("runda"), 0)),
                "seedStep": int(safe_float(row.get("pas_seed"), 0)),
                "newCandidates": int(safe_float(row.get("candidati_noi"), 0)),
                "promotedCandidates": int(safe_float(row.get("candidati_promovati"), 0)),
                "totalCandidates": int(safe_float(row.get("candidati_totali"), 0)),
                "bestScore": round(safe_float(row.get("scor_live_maxim"), 0.0), 3),
                "avgCost10mg": round(safe_float(row.get("cost_mediu_10mg_usd"), 0.0), 2),
                "minCost10mg": round(safe_float(row.get("cost_minim_10mg_usd"), 0.0), 2),
                "timestamp": _safe_str(row.get("timestamp")),
            }
        )

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if not molecules_df.empty:
        top = molecules_df.sort_values(["round", "live_rank_score"], ascending=[True, False]).head(60)
        for _, row in top.iterrows():
            smiles = _safe_str(row.get("smiles"))
            parent = _safe_str(row.get("parent_seed"))
            node_id = f"mol-{abs(hash(smiles))}"
            parent_id = f"mol-{abs(hash(parent))}" if parent else None
            nodes.append(
                {
                    "id": node_id,
                    "parentId": parent_id,
                    "label": _safe_str(row.get("transformare_afisare"), "molecula"),
                    "status": _safe_str(row.get("live_status")),
                    "round": int(safe_float(row.get("round"), 0)),
                    "rank": int(safe_float(row.get("rank"), 0)),
                    "score": round(safe_float(row.get("live_rank_score"), 0.0), 3),
                    "pic50": round(safe_float(row.get("predicted_pIC50"), 0.0), 3),
                    "deltaPic50": round(safe_float(row.get("delta_predicted_pIC50"), 0.0), 3),
                    "deltaScore": round(safe_float(row.get("delta_final_score"), 0.0), 3),
                }
            )
            if parent_id:
                edges.append({"source": parent_id, "target": node_id, "label": _safe_str(row.get("action_name"), "mutatie")})
    return {"generations": generations, "nodes": nodes, "edges": edges}


def _build_rl_monitor(molecules_df: pd.DataFrame, rounds_df: pd.DataFrame) -> dict[str, Any]:
    reward_series: list[dict[str, Any]] = []
    for _, row in rounds_df.iterrows():
        reward_series.append(
            {
                "round": int(safe_float(row.get("runda"), 0)),
                "bestScore": round(safe_float(row.get("scor_live_maxim"), 0.0), 3),
                "avgCost10mg": round(safe_float(row.get("cost_mediu_10mg_usd"), 0.0), 2),
                "timestamp": _safe_str(row.get("timestamp")),
            }
        )

    penalty_series: list[dict[str, Any]] = []
    if not molecules_df.empty:
        grouped = molecules_df.groupby("round", dropna=False)
        for round_idx, group in grouped:
            audit_pass_rate = group.get("audit_pass", pd.Series(dtype=object)).map(_boolish).mean()
            penalty_series.append(
                {
                    "round": int(safe_float(round_idx, 0)),
                    "toxicityPenalty": round(safe_float(group.get("structural_alert_count", pd.Series(dtype=float)).mean(), 0.0), 3),
                    "invalidPenalty": round(1.0 - safe_float(audit_pass_rate, 1.0), 3),
                    "uncertaintyPenalty": round(safe_float(group.get("uncertainty", pd.Series(dtype=float)).mean(), 0.0), 3),
                    "rewardRiskPenalty": round(safe_float(group.get("reward_hacking_risk", pd.Series(dtype=float)).mean(), 0.0), 3),
                    "verifiedReward": round(safe_float(group.get("verified_reward", pd.Series(dtype=float)).mean(), 0.0), 3),
                    "exploration": round(_clamp(1.0 - safe_float(group.get("parent_similarity", pd.Series(dtype=float)).mean(), 0.0)), 3),
                    "exploitation": round(_clamp(safe_float(group.get("generator_priority_score", pd.Series(dtype=float)).mean(), 0.0)), 3),
                }
            )

    return {
        "rewardSeries": reward_series,
        "penaltySeries": penalty_series,
        "verifiableNotes": [
            "Reward-ul afisat este derivat din campurile verificate din worker: verified_reward, uncertainty si reward_hacking_risk.",
            "Penalizarile sunt separate explicit pentru risc, incertitudine si validare chimica, astfel incat procesul sa nu fie black-box.",
        ],
    }


@lru_cache(maxsize=128)
def _molecule_view_payload(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "molBlock": "", "svg2d": "", "atomCount": 0, "formula": ""}

    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)
    svg = Draw.MolsToGridImage([mol_2d], molsPerRow=1, subImgSize=(420, 320), useSVG=True)
    svg_text = svg if isinstance(svg, str) else svg.data

    mol_3d = Chem.AddHs(Chem.Mol(mol))
    embed_status = AllChem.EmbedMolecule(mol_3d, randomSeed=0xF00D)
    if embed_status == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        except Exception:
            pass
    mol_block = Chem.MolToMolBlock(mol_3d if embed_status == 0 else mol_2d)
    svg_base64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")

    return {
        "smiles": smiles,
        "molBlock": mol_block,
        "svg2d": f"data:image/svg+xml;base64,{svg_base64}",
        "atomCount": int(mol.GetNumAtoms()),
        "formula": str(rdMolDescriptors.CalcMolFormula(mol)),
    }


def _build_detail(row: pd.Series | None, molecules_df: pd.DataFrame) -> dict[str, Any]:
    if row is None:
        return {"selected": None}

    ranking_breakdown = [
        {"label": "Baza generator", "value": round(safe_float(row.get("ranking_component_baza"), 0.0), 3), "tone": "positive"},
        {"label": "Noutate fata de piata", "value": round(safe_float(row.get("ranking_component_piata"), 0.0), 3), "tone": "positive"},
        {"label": "Ghidaj structural", "value": round(safe_float(row.get("ranking_component_structura"), 0.0), 3), "tone": "positive"},
        {"label": "Cost estimat", "value": round(safe_float(row.get("ranking_component_cost"), 0.0), 3), "tone": "positive"},
        {"label": "Fezabilitate sintetica", "value": round(safe_float(row.get("ranking_component_fezabilitate"), 0.0), 3), "tone": "positive"},
        {"label": "Certitudine", "value": round(safe_float(row.get("ranking_component_certitudine"), 0.0), 3), "tone": "positive"},
        {"label": "Penalizare risc", "value": round(safe_float(row.get("ranking_penalizare_risc"), 0.0), 3), "tone": "negative"},
    ]
    cost_breakdown = [
        {"label": "Cost 10 mg", "value": round(safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0), 2), "unit": "USD"},
        {"label": "Cost 100 mg", "value": round(safe_float(row.get("estimated_cost_for_100mg_usd"), 0.0), 2), "unit": "USD"},
        {"label": "Cost / mmol", "value": round(safe_float(row.get("estimated_cost_usd_per_mmol"), 0.0), 2), "unit": "USD"},
        {"label": "Pasi estimati", "value": round(safe_float(row.get("estimated_step_count"), 0.0), 2), "unit": "etape"},
        {"label": "Ore laborator", "value": round(safe_float(row.get("estimated_labor_hours"), 0.0), 2), "unit": "ore"},
        {"label": "Complexitate purificare", "value": round(safe_float(row.get("estimated_purification_complexity"), 0.0), 2), "unit": "indice"},
    ]
    agent_contributions = _agent_contributions(row)

    return {
        "selected": {
            "rank": int(safe_float(row.get("rank"), 0)),
            "smiles": _safe_str(row.get("smiles")),
            "status": _safe_str(row.get("live_status")),
            "score": round(safe_float(row.get("live_rank_score"), 0.0), 3),
            "round": int(safe_float(row.get("round"), 0)),
            "action": _safe_str(row.get("action_name")),
            "route": _safe_str(row.get("synthetic_route")),
            "parent": _safe_str(row.get("parent_seed")),
            "lineagePath": _safe_str(row.get("lineage_path")),
            "deltaPic50": round(safe_float(row.get("delta_predicted_pIC50"), 0.0), 3),
            "deltaQed": round(safe_float(row.get("delta_QED"), 0.0), 3),
            "deltaScore": round(safe_float(row.get("delta_final_score"), 0.0), 3),
            "cost10mg": round(safe_float(row.get("estimated_cost_for_10mg_usd"), 0.0), 2),
            "cost100mg": round(safe_float(row.get("estimated_cost_for_100mg_usd"), 0.0), 2),
            "marketReference": _safe_str(row.get("closest_market_name")),
            "marketSimilarity": round(safe_float(row.get("max_market_similarity"), 0.0), 3),
            "view": _molecule_view_payload(_safe_str(row.get("smiles"))),
            "metrics": _build_metrics(row),
            "agentContributions": agent_contributions,
            "rankingBreakdown": ranking_breakdown,
            "costBreakdown": cost_breakdown,
            "explainability": _build_explainability(row, agent_contributions),
            "admet": _build_admet(row),
            "decisionHistory": _build_decision_history(row, molecules_df),
        }
    }


def _build_agent_series(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []
    points: list[dict[str, Any]] = []
    grouped = molecules_df.groupby("round", dropna=False)
    for round_idx, group in grouped:
        components = [_agent_component_scores(pd.Series(row)) for row in group.to_dict(orient="records")]
        if not components:
            continue
        points.append(
            {
                "round": int(safe_float(round_idx, 0)),
                "generator": round(sum(item["generator"] for item in components) / len(components), 3),
                "toxicity": round(sum(item["toxicity"] for item in components) / len(components), 3),
                "validator": round(sum(item["validator"] for item in components) / len(components), 3),
                "optimizer": round(sum(item["optimizer"] for item in components) / len(components), 3),
            }
        )
    return points


def _build_ranking_stability(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []
    stability: list[dict[str, Any]] = []
    grouped = molecules_df.groupby("round", dropna=False)
    for round_idx, group in grouped:
        scores = pd.to_numeric(group.get("live_rank_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        promoted_rate = group.get("live_status", pd.Series(dtype=str)).fillna("").astype(str).eq("promovata").mean()
        stability.append(
            {
                "round": int(safe_float(round_idx, 0)),
                "topScore": round(safe_float(scores.max(), 0.0), 3),
                "meanScore": round(safe_float(scores.mean(), 0.0), 3),
                "promotedRate": round(safe_float(promoted_rate, 0.0), 3),
                "scoreSpread": round(max(0.0, safe_float(scores.max(), 0.0) - safe_float(scores.min(), 0.0)), 3),
            }
        )
    return stability


def _build_stage_gate_frame(molecules_df: pd.DataFrame) -> pd.DataFrame:
    if molecules_df.empty:
        return pd.DataFrame()

    audit_pass = molecules_df.get("audit_pass", pd.Series([False] * len(molecules_df), index=molecules_df.index)).map(_boolish)
    veto = molecules_df.get("veto", pd.Series([False] * len(molecules_df), index=molecules_df.index)).map(_boolish)
    hard_constraint_pass = molecules_df.get("hard_constraint_pass", pd.Series([True] * len(molecules_df), index=molecules_df.index)).map(_boolish)
    live_status = molecules_df.get("live_status", pd.Series([""] * len(molecules_df), index=molecules_df.index)).fillna("").astype(str)

    structural_guidance = pd.to_numeric(
        molecules_df.get("structural_guidance_score", pd.Series([0.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(0.0)
    synthetic_feasibility = pd.to_numeric(
        molecules_df.get("synthetic_feasibility_score", pd.Series([0.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(0.0)
    market_similarity = pd.to_numeric(
        molecules_df.get("max_market_similarity", pd.Series([1.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(1.0)
    market_novelty = pd.to_numeric(
        molecules_df.get("market_novelty_score", pd.Series([0.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(0.0)
    estimated_cost = pd.to_numeric(
        molecules_df.get("estimated_cost_for_10mg_usd", pd.Series([9999.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(9999.0)
    reward_risk = pd.to_numeric(
        molecules_df.get("reward_hacking_risk", pd.Series([1.0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(1.0)
    rounds = pd.to_numeric(
        molecules_df.get("round", pd.Series([0] * len(molecules_df), index=molecules_df.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    audit_chain = audit_pass & ~veto
    structural_chain = audit_chain & (structural_guidance >= 0.45)
    feasible_chain = structural_chain & hard_constraint_pass & (synthetic_feasibility >= 0.65)
    market_chain = feasible_chain & ((market_similarity <= 0.35) | (market_novelty >= 0.65))
    experimental_chain = market_chain & (estimated_cost <= 60.0) & (reward_risk <= 0.18)
    promoted_chain = experimental_chain & live_status.eq("promovata")

    return pd.DataFrame(
        {
            "round": rounds,
            "auditPass": audit_chain,
            "structuralReady": structural_chain,
            "feasible": feasible_chain,
            "marketReady": market_chain,
            "experimentalProxy": experimental_chain,
            "promoted": promoted_chain,
        }
    )


def _build_pipeline_stage_snapshots(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []

    stage_frame = _build_stage_gate_frame(molecules_df)
    total = max(1, len(stage_frame))
    stages = [
        ("Rankate", total, "Toate moleculele cu scor live in biblioteca curenta."),
        ("Audit", int(stage_frame["auditPass"].sum()), "Molecule care trec auditul si nu sunt veto."),
        ("Structura", int(stage_frame["structuralReady"].sum()), "Trec si filtrul de ghidaj structural disponibil live."),
        ("Fezab.", int(stage_frame["feasible"].sum()), "Au fezabilitate sintetica buna si constrangeri respectate."),
        ("Piata", int(stage_frame["marketReady"].sum()), "Raman suficient de noi fata de spatiul de piata."),
        ("Exp. proxy", int(stage_frame["experimentalProxy"].sum()), "Proxy live pentru readiness experimental: cost + risc + gates."),
        ("Promovate", int(stage_frame["promoted"].sum()), "Ajung in shortlist-ul live dupa lantul complet de filtre."),
    ]

    return [
        {"label": label, "count": count, "share": round(count / total, 3), "note": note}
        for label, count, note in stages
    ]


def _build_pipeline_progress(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    stage_frame = _build_stage_gate_frame(molecules_df)
    if stage_frame.empty:
        return []

    progress: list[dict[str, Any]] = []
    for round_idx, group in stage_frame.groupby("round", dropna=False):
        progress.append(
            {
                "round": int(safe_float(round_idx, 0)),
                "auditPass": int(group["auditPass"].sum()),
                "structuralReady": int(group["structuralReady"].sum()),
                "feasible": int(group["feasible"].sum()),
                "marketReady": int(group["marketReady"].sum()),
                "experimentalProxy": int(group["experimentalProxy"].sum()),
                "promoted": int(group["promoted"].sum()),
            }
        )
    return progress


def _build_maturation_series(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []

    stage_frame = _build_stage_gate_frame(molecules_df)
    enriched = molecules_df.copy()
    enriched = enriched.assign(
        experimental_proxy=stage_frame["experimentalProxy"].astype(float),
        verified_reward_numeric=pd.to_numeric(molecules_df.get("verified_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        synthetic_feasibility_numeric=pd.to_numeric(molecules_df.get("synthetic_feasibility_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        market_novelty_numeric=pd.to_numeric(molecules_df.get("market_novelty_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        structural_guidance_numeric=pd.to_numeric(molecules_df.get("structural_guidance_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        estimated_cost_score_numeric=pd.to_numeric(molecules_df.get("estimated_cost_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        reward_risk_numeric=pd.to_numeric(molecules_df.get("reward_hacking_risk", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        live_rank_score_numeric=pd.to_numeric(molecules_df.get("live_rank_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        predicted_pIC50_numeric=pd.to_numeric(molecules_df.get("predicted_pIC50", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
    )

    series: list[dict[str, Any]] = []
    for round_idx, group in enriched.groupby("round", dropna=False):
        cohort = group.sort_values(["live_rank_score_numeric", "predicted_pIC50_numeric"], ascending=[False, False]).head(10)
        if cohort.empty:
            continue

        series.append(
            {
                "round": int(safe_float(round_idx, 0)),
                "verifiedRewardScore": round(_clamp(safe_float(cohort["verified_reward_numeric"].mean(), 0.0) / 11.0), 3),
                "feasibility": round(_clamp(safe_float(cohort["synthetic_feasibility_numeric"].mean(), 0.0)), 3),
                "novelty": round(_clamp(safe_float(cohort["market_novelty_numeric"].mean(), 0.0)), 3),
                "structural": round(_clamp(safe_float(cohort["structural_guidance_numeric"].mean(), 0.0)), 3),
                "costScore": round(_clamp(safe_float(cohort["estimated_cost_score_numeric"].mean(), 0.0)), 3),
                "safety": round(_clamp(1.0 - safe_float(cohort["reward_risk_numeric"].mean(), 0.0)), 3),
                "experimentalReadyRate": round(_clamp(safe_float(cohort["experimental_proxy"].mean(), 0.0)), 3),
            }
        )
    return series


def _session_bottleneck(molecules_df: pd.DataFrame) -> str:
    if molecules_df.empty:
        return "Sesiunea nu are inca molecule generate."
    promoted_rate = molecules_df.get("live_status", pd.Series(dtype=str)).fillna("").astype(str).eq("promovata").mean()
    mean_uncertainty = safe_float(molecules_df.get("uncertainty", pd.Series(dtype=float)).mean(), 0.0)
    mean_cost = safe_float(molecules_df.get("estimated_cost_for_10mg_usd", pd.Series(dtype=float)).mean(), 0.0)
    mean_risk = safe_float(molecules_df.get("reward_hacking_risk", pd.Series(dtype=float)).mean(), 0.0)
    if promoted_rate < 0.12:
        return "Trierea este stricta si putine molecule trec in shortlist."
    if mean_uncertainty > 0.12:
        return "Modelele au incertitudine mare pe acest lot."
    if mean_cost > 45:
        return "Costul sintetic mediu este mai ridicat decat idealul."
    if mean_risk > 0.18:
        return "Guardrails si penalizarile de risc franeaza lotul."
    return "Fluxul este echilibrat; merita urmarite top moleculele."


def _build_session_compare(current_session_name: str | None) -> list[dict[str, Any]]:
    sessions_root = PROJECT_ROOT / "reports" / "gui_live"
    if not sessions_root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for session_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
        status, molecules_df, _ = _read_frames(session_dir)
        summary = _build_summary(status, molecules_df)
        rows.append(
            {
                "sessionName": session_dir.name,
                "modeLabel": _safe_str(status.get("mod_label"), _mode_label(_safe_str(status.get("mod"), "ghidat_ai"))),
                "statusLabel": _status_label(_safe_str(status.get("status"), "pregatire")),
                "moleculeCount": summary["moleculeCount"],
                "promotedCount": summary["promotedCount"],
                "bestPic50": summary["bestPic50"],
                "bestScore": summary["bestScore"],
                "meanCost10mg": round(safe_float(molecules_df.get("estimated_cost_for_10mg_usd", pd.Series(dtype=float)).mean(), 0.0), 2),
                "meanUncertainty": round(safe_float(molecules_df.get("uncertainty", pd.Series(dtype=float)).mean(), 0.0), 3),
                "bottleneck": _session_bottleneck(molecules_df),
                "updatedAt": _safe_str(status.get("updated_at")),
                "isCurrent": session_dir.name == current_session_name,
            }
        )

    rows.sort(key=lambda item: (not item["isCurrent"], item["updatedAt"]), reverse=False)
    return rows


def _build_experimental_planner(molecules_df: pd.DataFrame) -> list[dict[str, Any]]:
    if molecules_df.empty:
        return []
    shortlist = molecules_df.sort_values(["live_rank_score", "predicted_pIC50"], ascending=[False, False]).head(5)
    plans: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(shortlist.iterrows(), start=1):
        cost_100mg = safe_float(row.get("estimated_cost_for_100mg_usd"), 0.0)
        risk = safe_float(row.get("reward_hacking_risk"), 0.0)
        synth = safe_float(row.get("synthetic_feasibility_score"), 0.0)
        priority = "Prioritate 1" if index == 1 and risk < 0.18 and synth >= 0.65 else "Prioritate 2" if index <= 3 else "Rezerva"
        material_plan = "10 mg screening initial, apoi 100 mg confirmare" if cost_100mg <= 150 else "10 mg screening, apoi sinteza etapizata inainte de 100 mg"
        plans.append(
            {
                "smiles": _safe_str(row.get("smiles")),
                "rank": int(safe_float(row.get("rank"), 0)),
                "name": f"Candidat #{int(safe_float(row.get('rank'), 0))}",
                "priority": priority,
                "assay": "Test enzimatic EGFR kinaza, curba doza-raspuns in 10 puncte",
                "control": _safe_str(row.get("closest_market_name"), "Osimertinib"),
                "materialPlan": material_plan,
                "estimatedCost": round(cost_100mg, 2),
                "rationale": f"Scor {safe_float(row.get('live_rank_score'), 0.0):.2f} | pIC50 {safe_float(row.get('predicted_pIC50'), 0.0):.2f} | risc {risk:.2f} | sinteza {synth:.2f}",
                "route": _safe_str(row.get("synthetic_route")),
                "status": _safe_str(row.get("live_status")),
            }
        )
    return plans


def _build_analytics(molecules_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "agentSeries": _build_agent_series(molecules_df),
        "rankingStability": _build_ranking_stability(molecules_df),
        "pipelineStages": _build_pipeline_stage_snapshots(molecules_df),
        "pipelineProgress": _build_pipeline_progress(molecules_df),
        "maturationSeries": _build_maturation_series(molecules_df),
    }


def build_dashboard_payload(
    *,
    session_name: str | None = None,
    limit: int = 100,
    search: str = "",
    status_filter: str = "all",
    rank: int | None = None,
    smiles: str | None = None,
) -> dict[str, Any]:
    session_dir = resolve_session_dir(session_name)
    status, molecules_df, rounds_df = _read_frames(session_dir)
    selected_row = _select_row(molecules_df, rank=rank, smiles=smiles)
    selected_detail = _build_detail(selected_row, molecules_df)
    agents = selected_detail["selected"]["agentContributions"] if selected_detail.get("selected") else _agent_contributions(None)

    return {
        "overview": _build_overview(status, molecules_df, rounds_df),
        "agents": agents,
        "flows": _agent_flows(agents),
        "detail": selected_detail,
        "timeline": _build_timeline(molecules_df, rounds_df),
        "rlMonitor": _build_rl_monitor(molecules_df, rounds_df),
        "library": _build_library(molecules_df, limit=limit, search=search, status_filter=status_filter),
        "marketCompare": _build_market_compare(selected_row),
        "sessionCompare": _build_session_compare(session_dir.name),
        "experimentalPlanner": _build_experimental_planner(molecules_df),
        "analytics": _build_analytics(molecules_df),
        "logs": tail_text(session_dir / "worker.log", lines=160),
        "sources": status.get("source_urls", []),
    }


def build_molecule_payload(session_name: str | None, smiles: str) -> dict[str, Any]:
    session_dir = resolve_session_dir(session_name)
    _, molecules_df, _ = _read_frames(session_dir)
    selected_row = _select_row(molecules_df, rank=None, smiles=smiles)
    return _build_detail(selected_row, molecules_df)


def build_control_state(session_name: str | None = None) -> dict[str, Any]:
    session_dir = resolve_session_dir(session_name)
    status = _normalize_status_payload(safe_json(session_dir / "status.json"))
    return {
        "sessionName": session_dir.name,
        "sessionDir": str(session_dir),
        "running": _is_running_status(_safe_str(status.get("status"))),
        "pid": int(status.get("pid", 0) or 0),
        "status": _safe_str(status.get("status"), "pregatire"),
        "mode": _safe_str(status.get("mod"), "ghidat_ai"),
        "modeLabel": _safe_str(status.get("mod_label"), _mode_label(_safe_str(status.get("mod"), "ghidat_ai"))),
        "updatedAt": _safe_str(status.get("updated_at")),
    }


def build_sources_payload(session_name: str | None = None) -> dict[str, Any]:
    session_dir = resolve_session_dir(session_name)
    status = safe_json(session_dir / "status.json")
    formula_note = session_dir / "formula_cost_estimator.md"
    return {
        "sourceUrls": status.get("source_urls", []),
        "costModelNote": formula_note.read_text(encoding="utf-8") if formula_note.exists() else "",
    }
