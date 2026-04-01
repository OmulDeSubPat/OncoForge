from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT
from src.generation.run_generation_benchmark_suite import _backfill_missing_generation_metadata


ABLATION_DIR = PROJECT_ROOT / "reports" / "studii_ablatie"
TOP_KS_RANKING = [25, 50, 100, 250]
TOP_KS_FOCUSED = [10, 18, 25, 50]


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _first_existing_csv(*paths: Path) -> pd.DataFrame | None:
    for path in paths:
        df = _load_csv(path)
        if df is not None:
            return df
    return None


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def _status_rate(df: pd.DataFrame, column: str, value: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return float((df[column].astype(str) == value).mean())


def _evaluate_top_k(
    df: pd.DataFrame,
    *,
    study_name: str,
    strategy_name: str,
    score_column: str,
    top_k: int,
) -> dict[str, object]:
    ranked = df.sort_values(score_column, ascending=False).head(min(top_k, len(df))).copy()
    return {
        "study_name": study_name,
        "strategy": strategy_name,
        "score_column": score_column,
        "top_k": int(top_k),
        "n": int(len(ranked)),
        "mean_predicted_pIC50": float(_series(ranked, "predicted_pIC50").mean()) if not ranked.empty else 0.0,
        "mean_qed": float(_series(ranked, "QED").mean()) if not ranked.empty else 0.0,
        "mean_final_score": float(_series(ranked, "final_score").mean()) if not ranked.empty else 0.0,
        "mean_reward_hacking_risk": float(_series(ranked, "reward_hacking_risk").mean()) if not ranked.empty else 0.0,
        "mean_cross_database_consensus": float(_series(ranked, "cross_database_consensus_score").mean()) if not ranked.empty else 0.0,
        "mean_external_evidence_support": float(_series(ranked, "external_evidence_support").mean()) if not ranked.empty else 0.0,
        "mean_evidence_arbiter_support": float(_series(ranked, "evidence_arbiter_support").mean()) if not ranked.empty else 0.0,
        "mean_structure_evidence_support": float(_series(ranked, "structure_evidence_support").mean()) if not ranked.empty else 0.0,
        "mean_experimental_readiness": float(_series(ranked, "experimental_readiness_score").mean()) if not ranked.empty else 0.0,
        "mean_feasibility_score": float(_series(ranked, "feasibility_score").mean()) if not ranked.empty else 0.0,
        "mean_adaptive_action_prior": float(_series(ranked, "adaptive_action_prior").mean()) if not ranked.empty else 0.0,
        "mean_generator_priority_score": float(_series(ranked, "generator_priority_score").mean()) if not ranked.empty else 0.0,
        "mean_parent_improvement_count": float(_series(ranked, "parent_improvement_count").mean()) if not ranked.empty else 0.0,
        "audit_pass_rate": _status_rate(ranked, "audit_status", "pass"),
        "external_evidence_pass_rate": _status_rate(ranked, "external_evidence_status", "pass"),
        "evidence_arbiter_pass_rate": _status_rate(ranked, "evidence_arbiter_status", "pass"),
        "structure_evidence_pass_rate": _status_rate(ranked, "structure_evidence_status", "pass"),
        "ready_rate": _status_rate(ranked, "experimental_readiness_status", "ready"),
        "supporting_rate": _status_rate(ranked, "experimental_readiness_status", "supporting"),
        "cross_database_strong_rate": _status_rate(ranked, "cross_database_status", "strong"),
    }


def _plot_study(summary_df: pd.DataFrame, study_name: str, out_dir: Path) -> Path | None:
    plot_df = summary_df[
        (summary_df["study_name"] == study_name)
        & (summary_df["top_k"] == summary_df["top_k"].min())
    ].copy()
    if plot_df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = plot_df["strategy"].tolist()

    axes[0].bar(labels, plot_df["mean_predicted_pIC50"], color="#1d3557")
    axes[0].set_title("Potenta estimata medie")
    axes[0].set_ylabel("pIC50 mediu")
    axes[0].tick_params(axis="x", rotation=22)

    axes[1].bar(labels, plot_df["mean_reward_hacking_risk"], color="#e76f51")
    axes[1].set_title("Risc mediu de reward hacking")
    axes[1].set_ylabel("Scor mediu")
    axes[1].tick_params(axis="x", rotation=22)

    ready_metric = "ready_rate" if plot_df["ready_rate"].sum() > 0 else "audit_pass_rate"
    ready_label = "Rata ready" if ready_metric == "ready_rate" else "Rata audit pass"
    axes[2].bar(labels, plot_df[ready_metric], color="#2a9d8f")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(ready_label)
    axes[2].set_ylabel("Proportie")
    axes[2].tick_params(axis="x", rotation=22)

    fig.suptitle(f"Studiu de ablatie: {study_name.replace('_', ' ')}")
    fig.tight_layout()
    out_path = out_dir / f"{study_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _ranking_guardrail_ablation(ranked: pd.DataFrame) -> pd.DataFrame:
    df = ranked.copy()
    df["verified_plus_multiobiectiv"] = _series(df, "verified_reward") + 1.20 * _series(df, "multi_objective_score")
    df["fara_penalizare_risc"] = (
        _series(df, "verified_reward")
        + 1.20 * _series(df, "multi_objective_score")
        - 2.50 * _series(df, "veto")
        - _series(df, "audit_status_penalty")
    )
    df["fara_status_audit"] = (
        _series(df, "verified_reward")
        + 1.20 * _series(df, "multi_objective_score")
        - 1.50 * _series(df, "reward_hacking_risk")
        - 2.50 * _series(df, "veto")
    )
    df["fara_veto"] = (
        _series(df, "verified_reward")
        + 1.20 * _series(df, "multi_objective_score")
        - 1.50 * _series(df, "reward_hacking_risk")
        - _series(df, "audit_status_penalty")
    )

    strategies = [
        ("proxy_naiv", "naive_score"),
        ("verificat_plus_multiobiectiv", "verified_plus_multiobiectiv"),
        ("scor_protejat_final", "final_score"),
        ("fara_penalizare_risc", "fara_penalizare_risc"),
        ("fara_status_audit", "fara_status_audit"),
        ("fara_veto", "fara_veto"),
    ]

    rows: list[dict[str, object]] = []
    for strategy_name, score_column in strategies:
        for top_k in TOP_KS_RANKING:
            rows.append(
                _evaluate_top_k(
                    df,
                    study_name="ranking_guardrails",
                    strategy_name=strategy_name,
                    score_column=score_column,
                    top_k=top_k,
                )
            )
    return pd.DataFrame(rows)


def _readiness_component_ablation(readiness_df: pd.DataFrame) -> pd.DataFrame:
    df = readiness_df.copy()
    feasibility = _series(df, "feasibility_score")
    docking = _series(df, "docking_rescore")
    interaction = _series(df, "interaction_support_score")
    crossdb = _series(df, "cross_database_consensus_score")
    external = _series(df, "external_evidence_support")
    independent = (_series(df, "cross_database_independent_support_count") / 3.0).clip(lower=0.0, upper=1.0)
    market_alignment = _series(df, "market_alignment_support")
    active_support = _series(df, "max_active_similarity")
    source_support = _series(df, "source_support_score")
    traceability = _series(df, "traceability_score")
    qed = _series(df, "QED")
    synthetic = _series(df, "synthetic_ease_score")
    risk_support = (1.0 - _series(df, "reward_hacking_risk", 0.5)).clip(lower=0.0, upper=1.0)
    uncertainty = _series(df, "uncertainty", 0.20)
    uncertainty_scale = max(0.10, float(uncertainty.quantile(0.90))) if not uncertainty.empty else 0.20
    low_uncertainty = (1.0 - (uncertainty / uncertainty_scale)).clip(lower=0.0, upper=1.0)

    def readiness_score(
        *,
        use_crossdb: bool = True,
        use_external: bool = True,
        use_market: bool = True,
        use_traceability: bool = True,
    ) -> pd.Series:
        return (
            0.23 * feasibility
            + 0.14 * docking
            + 0.12 * interaction
            + (0.08 * crossdb if use_crossdb else 0.0)
            + (0.08 * external if use_external else 0.0)
            + (0.04 * independent if use_crossdb else 0.0)
            + (0.08 * market_alignment if use_market else 0.0)
            + 0.10 * active_support
            + 0.06 * source_support
            + (0.06 * traceability if use_traceability else 0.0)
            + 0.07 * qed
            + 0.03 * synthetic
            + 0.03 * risk_support
            + 0.04 * low_uncertainty
        ).clip(lower=0.0, upper=1.0)

    df["scor_readiness_complet"] = readiness_score()
    df["fara_consens_multisursa"] = readiness_score(use_crossdb=False)
    df["fara_dovezi_externe"] = readiness_score(use_external=False)
    df["fara_aliniere_piata"] = readiness_score(use_market=False)
    df["fara_trasabilitate"] = readiness_score(use_traceability=False)

    rows: list[dict[str, object]] = []
    for strategy_name, score_column in [
        ("scor_readiness_complet", "scor_readiness_complet"),
        ("fara_consens_multisursa", "fara_consens_multisursa"),
        ("fara_dovezi_externe", "fara_dovezi_externe"),
        ("fara_aliniere_piata", "fara_aliniere_piata"),
        ("fara_trasabilitate", "fara_trasabilitate"),
    ]:
        for top_k in TOP_KS_FOCUSED:
            rows.append(
                _evaluate_top_k(
                    df,
                    study_name="readiness_components",
                    strategy_name=strategy_name,
                    score_column=score_column,
                    top_k=top_k,
                )
            )
    return pd.DataFrame(rows)


def _generation_component_ablation(generation_df: pd.DataFrame) -> pd.DataFrame:
    df = generation_df.copy()
    df["scor_generare_complet"] = (
        _series(df, "final_score")
        + 0.95 * _series(df, "generator_priority_score")
        + 0.22 * _series(df, "adaptive_action_prior")
        + 0.25 * _series(df, "parent_similarity")
        + 0.12 * _series(df, "property_support_score")
        + 0.24 * _series(df, "structural_guidance_score")
        - 0.20 * _series(df, "reward_hacking_risk")
    )
    df["fara_prior_adaptiv"] = df["scor_generare_complet"] - 0.22 * _series(df, "adaptive_action_prior")
    df["fara_ghidare_structurala"] = df["scor_generare_complet"] - 0.24 * _series(df, "structural_guidance_score")
    df["fara_politica_generatorului"] = (
        _series(df, "final_score")
        + 0.25 * _series(df, "parent_similarity")
        - 0.20 * _series(df, "reward_hacking_risk")
    )

    rows: list[dict[str, object]] = []
    for strategy_name, score_column in [
        ("scor_generare_complet", "scor_generare_complet"),
        ("fara_prior_adaptiv", "fara_prior_adaptiv"),
        ("fara_ghidare_structurala", "fara_ghidare_structurala"),
        ("fara_politica_generatorului", "fara_politica_generatorului"),
    ]:
        for top_k in TOP_KS_FOCUSED:
            rows.append(
                _evaluate_top_k(
                    df,
                    study_name="generation_components",
                    strategy_name=strategy_name,
                    score_column=score_column,
                    top_k=top_k,
                )
            )
    return pd.DataFrame(rows)


def _rl_component_ablation(rl_df: pd.DataFrame) -> pd.DataFrame:
    df = rl_df.copy()
    base_column = "rl_priority_score"
    if base_column not in df.columns and "gpu_rl_priority_score" in df.columns:
        base_column = "gpu_rl_priority_score"
    if base_column not in df.columns and "actor_critic_priority_score" in df.columns:
        base_column = "actor_critic_priority_score"

    df["prioritate_rl_completa"] = _series(df, base_column)
    df["fara_readiness"] = df["prioritate_rl_completa"] - 0.25 * _series(df, "experimental_readiness_score")
    df["fara_dovezi_externe"] = df["prioritate_rl_completa"] - 0.30 * _series(df, "external_evidence_support")
    df["fara_arbitru_dovezi"] = df["prioritate_rl_completa"] - 0.25 * _series(df, "evidence_arbiter_support")
    df["fara_prior_adaptiv"] = df["prioritate_rl_completa"] - 0.16 * _series(df, "adaptive_action_prior", 0.5)

    rows: list[dict[str, object]] = []
    for strategy_name, score_column in [
        ("prioritate_rl_completa", "prioritate_rl_completa"),
        ("fara_readiness", "fara_readiness"),
        ("fara_dovezi_externe", "fara_dovezi_externe"),
        ("fara_arbitru_dovezi", "fara_arbitru_dovezi"),
        ("fara_prior_adaptiv", "fara_prior_adaptiv"),
    ]:
        for top_k in [5, 10, 20]:
            rows.append(
                _evaluate_top_k(
                    df,
                    study_name="rl_components",
                    strategy_name=strategy_name,
                    score_column=score_column,
                    top_k=top_k,
                )
            )
    return pd.DataFrame(rows)


def _load_generation_frame() -> pd.DataFrame | None:
    base_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"
    base_df = _load_csv(base_path)
    preferred = _first_existing_csv(
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_crossdb.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_crossdb.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_feasibility.csv",
        base_path,
    )
    if preferred is None:
        return base_df
    if base_df is not None and not base_df.empty and preferred is not base_df:
        return _backfill_missing_generation_metadata(preferred, base_df)
    return preferred


def _study_summary_lines(study_df: pd.DataFrame, study_name: str) -> list[str]:
    current_df = study_df[
        study_df["strategy"].str.contains("complet|final|protej", case=False, regex=True)
        & (study_df["top_k"] == study_df["top_k"].min())
    ]
    baseline_df = study_df[study_df["top_k"] == study_df["top_k"].min()].sort_values("mean_reward_hacking_risk")
    literature_df = _load_csv(PROJECT_ROOT / "comparatii_literatura.csv")
    literature_rows = 0 if literature_df is None else int(len(literature_df))

    lines = [f"## {study_name.replace('_', ' ').title()}"]
    if not current_df.empty:
        row = current_df.iloc[0]
        lines.extend(
            [
                "### Rezultatele modelului curent",
                f"- Strategia principala: `{row['strategy']}`",
                f"- Pentru top `{int(row['top_k'])}` candidati, potenta estimata medie este `{row['mean_predicted_pIC50']:.3f}`, iar riscul mediu de reward hacking este `{row['mean_reward_hacking_risk']:.3f}`.",
                f"- Rata `ready` este `{row['ready_rate']:.3f}`, iar rata `audit pass` este `{row['audit_pass_rate']:.3f}`.",
            ]
        )
    if not baseline_df.empty:
        best = baseline_df.iloc[0]
        worst = baseline_df.iloc[-1]
        lines.extend(
            [
                "### Comparatie cu baseline intern",
                f"- Cea mai sigura strategie din acest studiu este `{best['strategy']}`, cu risc mediu `{best['mean_reward_hacking_risk']:.3f}`.",
                f"- Strategia cea mai slaba pe acelasi criteriu este `{worst['strategy']}`, cu risc mediu `{worst['mean_reward_hacking_risk']:.3f}`.",
                "- Diferentele trebuie interpretate ca efecte de ranking intern, nu ca dovada experimentala.",
            ]
        )
    lines.extend(
        [
            "### Comparatie cu studii similare",
            (
                f"- Fisierul `comparatii_literatura.csv` contine `{literature_rows}` randuri, astfel comparatia externa ramane limitata."
                if literature_rows > 0
                else "- Comparatia cu literatura ramane provizorie deoarece `comparatii_literatura.csv` nu contine inca valori externe complete pentru acest studiu."
            ),
            "- In aceasta versiune, studiul de ablatie compara in primul rand variante interne ale pipeline-ului, conform artefactelor disponibile.",
            "",
        ]
    )
    return lines


def _write_markdown_report(study_frames: list[pd.DataFrame], out_dir: Path) -> None:
    report_path = out_dir / "rezumat_studii_ablatie.md"
    lines = [
        "# Studii de ablatie OncoForge",
        "",
        "Acest document sintetizeaza studii de ablatie interne pentru componentele-cheie ale pipeline-ului.",
        "Rezultatele descriu comportamentul in silico al strategiilor alternative si nu inlocuiesc validarea experimentala.",
        "",
    ]
    for study_df in study_frames:
        if study_df.empty:
            continue
        study_name = str(study_df["study_name"].iloc[0])
        lines.extend(_study_summary_lines(study_df, study_name))
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ruleaza un pachet de studii de ablatie pentru OncoForge.")
    parser.add_argument("--out-dir", type=str, default=str(ABLATION_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = _load_csv(PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv")
    readiness = _first_existing_csv(
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_readiness.csv",
        PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates_structural_feasibility.csv",
    )
    generation = _load_generation_frame()
    rl_df = _load_csv(PROJECT_ROOT / "reports" / "rl_verifiable" / "rl_top_candidates.csv")

    if ranked is None or ranked.empty:
        raise FileNotFoundError("Lipseste `reports/ranked_egfr_dataset.csv` pentru studiul de ablatie.")

    study_frames: list[pd.DataFrame] = []
    study_frames.append(_ranking_guardrail_ablation(ranked))
    if readiness is not None and not readiness.empty:
        study_frames.append(_readiness_component_ablation(readiness))
    if generation is not None and not generation.empty:
        study_frames.append(_generation_component_ablation(generation))
    if rl_df is not None and not rl_df.empty:
        study_frames.append(_rl_component_ablation(rl_df))
    if not study_frames:
        raise RuntimeError("Nu exista artefacte suficiente pentru a construi studiile de ablatie.")

    combined = pd.concat(study_frames, ignore_index=True, sort=False)
    combined.to_csv(out_dir / "studii_ablatie.csv", index=False)
    (out_dir / "studii_ablatie.json").write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    plots: dict[str, str] = {}
    for study_df in study_frames:
        if study_df.empty:
            continue
        study_name = str(study_df["study_name"].iloc[0])
        plot_path = _plot_study(combined, study_name, out_dir)
        if plot_path is not None:
            plots[study_name] = str(plot_path)

    (out_dir / "studii_ablatie_plots.json").write_text(json.dumps(plots, indent=2), encoding="utf-8")
    _write_markdown_report(study_frames, out_dir)

    print(f"[OK] Salvate studii de ablatie in: {out_dir}")
    print(combined.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
