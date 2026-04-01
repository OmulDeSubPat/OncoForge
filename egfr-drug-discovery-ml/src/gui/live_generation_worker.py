from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_BOOTSTRAP_ROOT))

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.economics.cost_model import LITERATURE_SOURCE_URLS, add_cost_estimates, build_cost_model_markdown
from src.generation.generation_benchmark import summarize_generated_frame
from src.generation.lineage_tracking import add_parent_child_tracking
from src.generation.medchem_mutations import generate_medchem_outcomes
from src.pipelines.artifact_utils import load_csv_artifact
from src.utils.chem import canonicalize_smiles


MODE_LABELS = {
    "explorare": "Explorare rapida",
    "ghidat_ai": "Generare ghidata de AI",
    "iterativ": "Optimizare iterativa",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _ensure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"gui_live_worker_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stdout_handler)
    return logger


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def _write_status(session_dir: Path, status: str, **extra: object) -> None:
    payload: dict[str, object] = {
        "status": status,
        "updated_at": _now_iso(),
        "pid": os.getpid(),
        "session_dir": str(session_dir),
        "source_urls": LITERATURE_SOURCE_URLS,
    }
    payload.update(extra)
    _write_json(session_dir / "status.json", payload)


def _clear_previous_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "status.json",
        "molecule_generate.csv",
        "molecule_generate.summary.json",
        "molecule_generate_backup.csv",
        "rezumat_runde.csv",
        "rezumat_runde_backup.csv",
        "worker.log",
        "formula_cost_estimator.md",
    ]:
        path = session_dir / filename
        if path.exists():
            path.unlink()


def _save_formula_note(session_dir: Path) -> None:
    (session_dir / "formula_cost_estimator.md").write_text(build_cost_model_markdown(), encoding="utf-8")


def _select_seed_pool(df: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    if mode == "explorare":
        filtered = df[
            (df["veto"] == False)
            & (df["audit_pass"] == True)
            & (df["reward_hacking_risk"] <= 0.35)
            & (df["agent_disagreement_score"] <= 0.45)
            & (_series(df, "applicability_score", 0.0) >= 0.30)
        ].copy()
    elif mode == "ghidat_ai":
        filtered = df[
            (df["predicted_pIC50"] >= 8.5)
            & (df["QED"] >= 0.40)
            & (df["reward_hacking_risk"] <= 0.30)
            & (df["agent_disagreement_score"] <= 0.45)
            & (df["audit_pass"] == True)
            & (df["veto"] == False)
        ].copy()
    else:
        filtered = df[
            (df["predicted_pIC50"] >= 8.5)
            & (df["QED"] >= 0.40)
            & (df["reward_hacking_risk"] <= 0.30)
            & (df["agent_disagreement_score"] <= 0.45)
            & (_series(df, "applicability_score", 0.0) >= 0.30)
            & (df["audit_pass"] == True)
            & (df["veto"] == False)
        ].copy()

    current_pool = filtered.sort_values("final_score", ascending=False).head(int(top_k)).reset_index(drop=True)
    current_pool["ancestor_seed"] = current_pool["smiles"]
    current_pool["lineage_depth"] = 0
    current_pool["lineage_path"] = current_pool["smiles"]
    return current_pool


def _generate_candidate_rows(
    parent_row: pd.Series,
    *,
    round_idx: int,
    variants_per_seed: int,
    seen: set[str],
) -> tuple[list[dict[str, Any]], int]:
    parent_smiles = str(parent_row["smiles"])
    ancestor_seed = str(parent_row.get("ancestor_seed", parent_smiles))
    lineage_depth = int(parent_row.get("lineage_depth", 0) or 0)
    lineage_path = str(parent_row.get("lineage_path", parent_smiles))
    variants = generate_medchem_outcomes(parent_smiles, max_variants=int(variants_per_seed))

    candidate_rows: list[dict[str, Any]] = []
    for variant in variants:
        canonical_smiles = canonicalize_smiles(variant.smiles)
        if not canonical_smiles or canonical_smiles in seen:
            continue
        seen.add(canonical_smiles)
        candidate_rows.append(
            {
                "smiles": canonical_smiles,
                "parent_seed": parent_smiles,
                "ancestor_seed": ancestor_seed,
                "lineage_depth": lineage_depth + 1,
                "lineage_path": f"{lineage_path} -> {canonical_smiles}",
                "round": round_idx,
                "action_name": variant.action_name,
                "action_category": variant.category,
                "action_rule_source": variant.rule_source,
                "reaction_family": variant.reaction_family,
                "synthetic_route": variant.synthetic_route,
                "synthetic_feasibility_score": variant.synthetic_feasibility_score,
                "medchem_realism_score": variant.medchem_realism_score,
                "transformation_confidence_score": variant.transformation_confidence,
                "preserves_scaffold": variant.preserves_scaffold,
                "parent_similarity": variant.parent_similarity,
                "property_support_score": variant.property_support_score,
                "category_priority_score": variant.category_priority_score,
                "generator_priority_score": variant.generator_priority_score,
                "adaptive_action_prior": variant.adaptive_action_prior,
                "hard_constraint_pass": variant.hard_constraint_pass,
                "hard_constraint_notes": variant.hard_constraint_notes,
                "introduced_warhead": variant.introduced_warhead,
                "warhead_retained": variant.warhead_retained,
                "alert_count": variant.alert_count,
                "severe_alert_count": variant.severe_alert_count,
                "structural_guidance_score": variant.structural_guidance_score,
                "structure_guidance_reference": variant.structure_guidance_reference,
                "structure_guidance_backend": variant.structure_guidance_backend,
            }
        )
    return candidate_rows, len(variants)


def _compute_generation_score(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "explorare":
        out["generator_composite_score"] = (
            _series(out, "final_score")
            + 0.70 * _series(out, "generator_priority_score")
            + 0.20 * _series(out, "parent_similarity")
            + 0.10 * _series(out, "property_support_score")
            + 0.20 * _series(out, "structural_guidance_score")
            + 0.18 * _series(out, "adaptive_action_prior")
            - 0.25 * _series(out, "reward_hacking_risk")
        )
    elif mode == "ghidat_ai":
        out["generator_composite_score"] = (
            _series(out, "final_score")
            + 0.80 * _series(out, "generator_priority_score")
            + 0.18 * _series(out, "adaptive_action_prior")
            + 0.20 * _series(out, "parent_similarity")
            + 0.10 * _series(out, "property_support_score")
            + 0.22 * _series(out, "structural_guidance_score")
            - 0.20 * _series(out, "reward_hacking_risk")
        )
    else:
        out["generator_composite_score"] = (
            _series(out, "final_score")
            + 0.95 * _series(out, "generator_priority_score")
            + 0.22 * _series(out, "adaptive_action_prior")
            + 0.25 * _series(out, "parent_similarity")
            + 0.12 * _series(out, "property_support_score")
            + 0.24 * _series(out, "structural_guidance_score")
            - 0.20 * _series(out, "reward_hacking_risk")
        )

    out["market_novelty_score"] = 1.0 - _series(out, "max_market_similarity")
    out["ranking_component_baza"] = _series(out, "generator_composite_score")
    out["ranking_component_piata"] = 0.30 * _series(out, "market_novelty_score")
    out["ranking_component_structura"] = 0.18 * _series(out, "structural_guidance_score")
    out["ranking_component_cost"] = 0.35 * _series(out, "estimated_cost_score")
    out["ranking_component_fezabilitate"] = 0.18 * _series(out, "synthetic_feasibility_score")
    out["ranking_component_certitudine"] = 0.10 * (1.0 - _series(out, "uncertainty").clip(lower=0.0, upper=0.35) / 0.35)
    out["ranking_penalizare_risc"] = (
        0.30 * _series(out, "reward_hacking_risk")
        + 0.12 * _series(out, "agent_disagreement_score")
    )
    out["live_rank_score"] = (
        _series(out, "ranking_component_baza")
        + _series(out, "ranking_component_piata")
        + _series(out, "ranking_component_structura")
        + _series(out, "ranking_component_cost")
        + _series(out, "ranking_component_fezabilitate")
        + _series(out, "ranking_component_certitudine")
        - _series(out, "ranking_penalizare_risc")
    )
    return out


def _promotion_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "explorare":
        return (
            (_series(df, "predicted_pIC50") >= 8.0)
            & (_series(df, "QED") >= 0.30)
            & (_series(df, "reward_hacking_risk") <= 0.45)
            & (_series(df, "agent_disagreement_score") <= 0.55)
            & (_series(df, "generator_priority_score") >= 0.25)
            & (df["audit_status"].fillna("pass") != "fail")
            & (df["veto"] == False)
        )
    if mode == "ghidat_ai":
        return (
            (_series(df, "generator_priority_score") >= 0.35)
            & (_series(df, "predicted_pIC50") >= 8.2)
            & (_series(df, "QED") >= 0.35)
            & (_series(df, "reward_hacking_risk") <= 0.45)
            & (df["audit_status"].fillna("pass") != "fail")
            & (df["veto"] == False)
        )
    return (
        (_series(df, "predicted_pIC50") >= 8.3)
        & (_series(df, "QED") >= 0.35)
        & (_series(df, "reward_hacking_risk") <= 0.45)
        & (_series(df, "agent_disagreement_score") <= 0.55)
        & (_series(df, "generator_priority_score") >= 0.40)
        & (df["audit_status"].fillna("pass") != "fail")
        & (df["veto"] == False)
    )


def _apply_candidate_status(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    promoted = _promotion_mask(out, mode)
    out["live_status"] = "revizie"
    out.loc[promoted, "live_status"] = "promovata"
    out.loc[(out["audit_pass"] == False) | (out["veto"] == True) | (out["audit_status"].fillna("pass") == "fail"), "live_status"] = "respinsa"
    return out


def _merge_cumulative_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["live_rank_score", "generator_composite_score", "final_score", "predicted_pIC50"],
        ascending=[False, False, False, False],
    ).drop_duplicates(subset=["smiles"], keep="first")
    out = out.reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def _build_current_pool(df: pd.DataFrame, beam_width: int) -> pd.DataFrame:
    promoted = df[df["live_status"] == "promovata"].copy()
    if promoted.empty:
        return pd.DataFrame(columns=df.columns)
    return promoted.sort_values(
        ["live_rank_score", "generator_composite_score", "final_score"],
        ascending=[False, False, False],
    ).head(int(beam_width)).reset_index(drop=True)


def _persist_snapshot(
    *,
    session_dir: Path,
    cumulative_df: pd.DataFrame,
    event_rows: list[dict[str, object]],
) -> None:
    cumulative_df.to_csv(session_dir / "molecule_generate.csv", index=False)
    pd.DataFrame(event_rows).to_csv(session_dir / "rezumat_runde.csv", index=False)


def run_session(
    session_dir: Path,
    *,
    mode: str = "iterativ",
    seed_count: int = 8,
    rounds: int = 3,
    variants_per_seed: int = 30,
    beam_width: int = 8,
    sleep_seconds: float = 0.15,
    reset_session: bool = True,
) -> None:
    if reset_session:
        _clear_previous_session(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    _save_formula_note(session_dir)

    logger = _ensure_logger(session_dir / "worker.log")
    logger.info("Pornesc sesiunea live in %s", session_dir)

    ranked_path = PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"
    ranked_df = load_csv_artifact(
        ranked_path,
        required_columns=[
            "smiles",
            "predicted_pIC50",
            "QED",
            "reward_hacking_risk",
            "agent_disagreement_score",
            "applicability_score",
            "audit_pass",
            "veto",
            "final_score",
        ],
        producer="python -m src.models.rank_dataset",
    )

    scorer = build_default_scorer()
    current_pool = _select_seed_pool(ranked_df, mode=mode, top_k=seed_count)
    actual_rounds = int(rounds if mode == "iterativ" else 1)
    seen = set(current_pool["smiles"].tolist())
    cumulative_batches: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    total_attempted = 0
    total_new_unique = 0

    _write_status(
        session_dir,
        "pornit",
        mesaj="Worker-ul a pornit",
        mod=mode,
        mod_label=MODE_LABELS.get(mode, mode),
        total_rounds=actual_rounds,
        seed_count=len(current_pool),
    )

    try:
        for round_idx in range(1, actual_rounds + 1):
            if current_pool.empty:
                logger.info("Nu mai exista molecule parinte pentru runda %s.", round_idx)
                break

            logger.info("Runda %s/%s | parinti activi: %s", round_idx, actual_rounds, len(current_pool))
            round_promoted_batches: list[pd.DataFrame] = []
            round_seed_count = len(current_pool)

            for seed_idx, (_, parent_row) in enumerate(current_pool.iterrows(), start=1):
                parent_smiles = str(parent_row["smiles"])
                candidate_rows, attempted = _generate_candidate_rows(
                    parent_row,
                    round_idx=round_idx,
                    variants_per_seed=variants_per_seed,
                    seen=seen,
                )
                total_attempted += int(attempted)
                total_new_unique += len(candidate_rows)

                if not candidate_rows:
                    event_rows.append(
                        {
                            "runda": round_idx,
                            "pas_seed": seed_idx,
                            "parinte": parent_smiles,
                            "candidati_noi": 0,
                            "candidati_promovati": 0,
                            "candidati_totali": len(_merge_cumulative_frames(cumulative_batches)),
                            "scor_live_maxim": None,
                            "cost_minim_10mg_usd": None,
                            "cost_mediu_10mg_usd": None,
                            "timestamp": _now_iso(),
                        }
                    )
                    _persist_snapshot(
                        session_dir=session_dir,
                        cumulative_df=_merge_cumulative_frames(cumulative_batches),
                        event_rows=event_rows,
                    )
                    _write_status(
                        session_dir,
                        "in_rulare",
                        mesaj="Fara candidati noi pentru parintele curent",
                        mod=mode,
                        mod_label=MODE_LABELS.get(mode, mode),
                        current_round=round_idx,
                        total_rounds=actual_rounds,
                        current_seed=seed_idx,
                        total_seeds=round_seed_count,
                        molecule_count=len(_merge_cumulative_frames(cumulative_batches)),
                        attempted_candidates=total_attempted,
                        generated_candidates=total_new_unique,
                        last_parent=parent_smiles,
                    )
                    continue

                candidate_df = pd.DataFrame(candidate_rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
                scored = score_smiles_list(candidate_df["smiles"].tolist(), scorer=scorer)
                scored = scored.merge(candidate_df, on="smiles", how="left")
                scored = add_parent_child_tracking(scored, parent_reference=ranked_df)
                scored = add_cost_estimates(scored)
                scored = _compute_generation_score(scored, mode=mode)
                scored = _apply_candidate_status(scored, mode=mode)
                scored = scored.sort_values(
                    ["live_rank_score", "generator_composite_score", "final_score", "predicted_pIC50"],
                    ascending=[False, False, False, False],
                ).reset_index(drop=True)

                promoted = _build_current_pool(scored, beam_width=beam_width)
                cumulative_batches.append(scored)
                if not promoted.empty:
                    round_promoted_batches.append(promoted)

                cumulative_df = _merge_cumulative_frames(cumulative_batches)
                event_rows.append(
                    {
                        "runda": round_idx,
                        "pas_seed": seed_idx,
                        "parinte": parent_smiles,
                        "candidati_noi": len(scored),
                        "candidati_promovati": len(promoted),
                        "candidati_totali": len(cumulative_df),
                        "scor_live_maxim": float(cumulative_df["live_rank_score"].max()) if not cumulative_df.empty else None,
                        "cost_minim_10mg_usd": float(cumulative_df["estimated_cost_for_10mg_usd"].min()) if not cumulative_df.empty else None,
                        "cost_mediu_10mg_usd": float(cumulative_df["estimated_cost_for_10mg_usd"].mean()) if not cumulative_df.empty else None,
                        "timestamp": _now_iso(),
                    }
                )
                _persist_snapshot(session_dir=session_dir, cumulative_df=cumulative_df, event_rows=event_rows)

                best_row = cumulative_df.iloc[0].to_dict() if not cumulative_df.empty else {}
                _write_status(
                    session_dir,
                    "in_rulare",
                    mesaj="Generarea ruleaza",
                    mod=mode,
                    mod_label=MODE_LABELS.get(mode, mode),
                    current_round=round_idx,
                    total_rounds=actual_rounds,
                    current_seed=seed_idx,
                    total_seeds=round_seed_count,
                    molecule_count=len(cumulative_df),
                    attempted_candidates=total_attempted,
                    generated_candidates=total_new_unique,
                    best_smiles=best_row.get("smiles"),
                    best_score=float(best_row.get("live_rank_score", 0.0) or 0.0),
                    best_cost_10mg_usd=float(best_row.get("estimated_cost_for_10mg_usd", 0.0) or 0.0),
                    last_parent=parent_smiles,
                )

                time.sleep(max(0.0, sleep_seconds))

            if mode == "iterativ":
                if round_promoted_batches:
                    current_pool = _build_current_pool(
                        _merge_cumulative_frames(round_promoted_batches),
                        beam_width=beam_width,
                    )
                else:
                    current_pool = pd.DataFrame(columns=current_pool.columns)

        final_df = _merge_cumulative_frames(cumulative_batches)
        if final_df.empty:
            _write_status(
                session_dir,
                "finalizat",
                mesaj="Nu au fost generate molecule noi",
                mod=mode,
                mod_label=MODE_LABELS.get(mode, mode),
                molecule_count=0,
                attempted_candidates=total_attempted,
                generated_candidates=total_new_unique,
            )
            return

        _persist_snapshot(session_dir=session_dir, cumulative_df=final_df, event_rows=event_rows)
        summarize_generated_frame(
            final_df,
            benchmark_name=f"gui_live_generation_{mode}",
            out_path=session_dir / "molecule_generate.summary.json",
            extra={
                "mode": mode,
                "seed_count": int(seed_count),
                "rounds": int(actual_rounds),
                "variants_per_seed": int(variants_per_seed),
                "beam_width": int(beam_width),
                "attempted_candidates": int(total_attempted),
                "generated_candidates": int(total_new_unique),
            },
        )

        best_row = final_df.iloc[0].to_dict()
        logger.info("Sesiunea s-a incheiat cu %s molecule distincte.", len(final_df))
        _write_status(
            session_dir,
            "finalizat",
            mesaj="Sesiune finalizata cu succes",
            mod=mode,
            mod_label=MODE_LABELS.get(mode, mode),
            molecule_count=len(final_df),
            attempted_candidates=total_attempted,
            generated_candidates=total_new_unique,
            best_smiles=best_row.get("smiles"),
            best_score=float(best_row.get("live_rank_score", 0.0) or 0.0),
            best_cost_10mg_usd=float(best_row.get("estimated_cost_for_10mg_usd", 0.0) or 0.0),
        )
    except Exception as exc:
        logger.exception("Worker-ul live a esuat")
        _write_status(
            session_dir,
            "eroare",
            mesaj=str(exc),
            mod=mode,
            mod_label=MODE_LABELS.get(mode, mode),
            attempted_candidates=total_attempted,
            generated_candidates=total_new_unique,
        )
        raise
    finally:
        working_csv = session_dir / "molecule_generate.csv"
        rounds_csv = session_dir / "rezumat_runde.csv"
        if working_csv.exists():
            shutil.copy2(working_csv, session_dir / "molecule_generate_backup.csv")
        if rounds_csv.exists():
            shutil.copy2(rounds_csv, session_dir / "rezumat_runde_backup.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Worker live pentru generare si ranking de molecule.")
    parser.add_argument(
        "--session-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "gui_live" / "sesiune_curenta"),
        help="Directorul sesiunii live.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="iterativ",
        choices=sorted(MODE_LABELS.keys()),
        help="Modul de generare.",
    )
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--variants-per-seed", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--no-reset-session", action="store_true")
    args = parser.parse_args(argv)

    run_session(
        Path(args.session_dir),
        mode=args.mode,
        seed_count=args.seed_count,
        rounds=args.rounds,
        variants_per_seed=args.variants_per_seed,
        beam_width=args.beam_width,
        sleep_seconds=args.sleep_seconds,
        reset_session=not args.no_reset_session,
    )


if __name__ == "__main__":
    main()
