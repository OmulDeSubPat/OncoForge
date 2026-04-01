from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

PROJECT_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_BOOTSTRAP_ROOT))

from src.config import PROJECT_ROOT
from src.economics.cost_model import LITERATURE_SOURCE_URLS, build_cost_model_markdown
from src.gui.data_access import prepare_molecule_frame, safe_csv, safe_json, safe_float, summarize_molecules, tail_text
from src.gui.labels import HELP_LINES, TAB_NAMES, mode_note, status_label, status_note
from src.gui.live_generation_worker import MODE_LABELS
from src.gui.theme import inject_global_theme, render_header_banner, render_info_strip, render_status_banner


SESSION_DIR = PROJECT_ROOT / "reports" / "gui_live" / "sesiune_curenta"
GUI_ROOT = PROJECT_ROOT / "reports" / "gui_live"


def _load_streamlit():
    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Streamlit nu este instalat. Ruleaza `pip install -r requirements.txt` si apoi `streamlit run src/gui/chemist_dashboard.py`."
        ) from exc
    return st


def _render_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(560, 380))


def _launch_worker(
    *,
    session_dir: Path,
    mode: str,
    seed_count: int,
    rounds: int,
    variants_per_seed: int,
    beam_width: int,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.gui.live_generation_worker",
        "--session-dir",
        str(session_dir),
        "--mode",
        str(mode),
        "--seed-count",
        str(seed_count),
        "--rounds",
        str(rounds),
        "--variants-per-seed",
        str(variants_per_seed),
        "--beam-width",
        str(beam_width),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _terminate_worker(pid: int) -> None:
    if pid <= 0:
        return
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=creationflags,
        )
    else:  # pragma: no cover
        import os
        import signal

        os.kill(pid, signal.SIGTERM)


def _running_status(status: dict[str, object]) -> bool:
    return str(status.get("status", "") or "").lower() == "in_rulare"


def _reset_session_dir(session_dir: Path) -> None:
    resolved = session_dir.resolve()
    allowed_root = GUI_ROOT.resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise ValueError("Directorul sesiunii trebuie sa fie in reports/gui_live.")
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)


def _progress_fraction(status: dict[str, object]) -> float:
    total_rounds = max(1, int(status.get("total_rounds", 1) or 1))
    total_seeds = max(1, int(status.get("total_seeds", 1) or 1))
    current_round = int(status.get("current_round", 0) or 0)
    current_seed = int(status.get("current_seed", 0) or 0)
    return min(1.0, ((max(0, current_round - 1) * total_seeds) + current_seed) / (total_rounds * total_seeds))


def _filter_molecule_frame(st, molecules_df: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    control_cols = st.columns([1.15, 1.0, 1.0])
    available_states = sorted(molecules_df["stare_afisare"].dropna().unique().tolist())
    default_states = [state for state in ["Promitatoare", "Necesita revizie"] if state in available_states]
    selected_states = control_cols[0].multiselect(
        "Stari afisate",
        options=available_states,
        default=default_states or available_states,
        key=f"{key_prefix}_states",
    )
    search_text = control_cols[1].text_input(
        "Cauta dupa SMILES sau transformare",
        key=f"{key_prefix}_search",
    ).strip().lower()
    top_limit = int(control_cols[2].slider("Numar randuri", 10, 150, 60, 10, key=f"{key_prefix}_limit"))

    filtered = molecules_df.copy()
    if selected_states:
        filtered = filtered[filtered["stare_afisare"].isin(selected_states)]

    if "predicted_pIC50" in filtered.columns and not filtered["predicted_pIC50"].dropna().empty:
        min_val = float(filtered["predicted_pIC50"].min())
        max_val = float(filtered["predicted_pIC50"].max())
        low = round(math.floor(min_val * 10) / 10, 1)
        high = round(math.ceil(max_val * 10) / 10, 1)
        pic50_range = st.slider("Filtru pIC50 prezis", low, high, (low, high), 0.1, key=f"{key_prefix}_pic50")
        filtered = filtered[
            (pd.to_numeric(filtered["predicted_pIC50"], errors="coerce") >= pic50_range[0])
            & (pd.to_numeric(filtered["predicted_pIC50"], errors="coerce") <= pic50_range[1])
        ]

    if search_text:
        filtered = filtered[
            filtered["smiles"].fillna("").astype(str).str.lower().str.contains(search_text, na=False)
            | filtered["transformare_afisare"].fillna("").astype(str).str.lower().str.contains(search_text, na=False)
        ]

    return filtered.sort_values(["live_rank_score", "predicted_pIC50", "QED"], ascending=[False, False, False]).head(top_limit)


def _render_metric_row(st, summary: dict[str, float]) -> None:
    metrics = st.columns(6)
    metrics[0].metric("Molecule evaluate", int(summary["molecule_count"]))
    metrics[1].metric("Promitatoare", int(summary["promising_count"]))
    metrics[2].metric("In revizie", int(summary["review_count"]))
    metrics[3].metric("Respinse", int(summary["rejected_count"]))
    metrics[4].metric("pIC50 maxim", f"{summary['best_pic50']:.2f}")
    metrics[5].metric("QED mediu", f"{summary['mean_qed']:.2f}")


def _render_general_tab(st, molecules_df: pd.DataFrame, rounds_df: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Evolutia scorului live")
        if {"timestamp", "scor_live_maxim"}.issubset(rounds_df.columns):
            chart_df = rounds_df[["timestamp", "scor_live_maxim"]].copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
            chart_df = chart_df.dropna().set_index("timestamp")
            if not chart_df.empty:
                st.line_chart(chart_df)
            else:
                st.info("Scorul live va aparea dupa primele evenimente.")
        else:
            st.info("Nu exista inca suficiente evenimente pentru acest grafic.")

    with right:
        st.subheader("Distributia moleculelor dupa stare")
        if "live_status" in molecules_df.columns and not molecules_df.empty:
            counts = molecules_df["live_status"].fillna("necunoscut").value_counts().rename(index=status_label)
            st.bar_chart(counts)
        else:
            st.info("Distributia starilor apare dupa primul lot evaluat.")

    left, right = st.columns(2)
    with left:
        st.subheader("pIC50 vs QED")
        if {"predicted_pIC50", "QED"}.issubset(molecules_df.columns):
            scatter_df = molecules_df[["predicted_pIC50", "QED"]].copy()
            scatter_df.columns = ["pIC50 prezis", "QED"]
            st.scatter_chart(scatter_df, x="pIC50 prezis", y="QED")
        else:
            st.info("Graficul apare dupa primul lot de molecule.")

    with right:
        st.subheader("Shortlist rapid")
        shortlist = molecules_df.sort_values(["live_rank_score", "predicted_pIC50"], ascending=[False, False]).head(8).copy()
        if shortlist.empty:
            st.info("Shortlist-ul apare dupa primul lot de molecule.")
        else:
            view = shortlist[
                [
                    column
                    for column in ["rank", "stare_afisare", "transformare_afisare", "predicted_pIC50", "QED", "synthetic_feasibility_score"]
                    if column in shortlist.columns
                ]
            ].rename(
                columns={
                    "rank": "Rang",
                    "stare_afisare": "Stare",
                    "transformare_afisare": "Transformare",
                    "predicted_pIC50": "pIC50 prezis",
                    "QED": "QED",
                    "synthetic_feasibility_score": "Fezabilitate",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("Indicatori secundari: cost si fezabilitate", expanded=False):
        left, right = st.columns(2)
        with left:
            if {"timestamp", "cost_mediu_10mg_usd", "cost_minim_10mg_usd"}.issubset(rounds_df.columns):
                chart_df = rounds_df[["timestamp", "cost_mediu_10mg_usd", "cost_minim_10mg_usd"]].copy()
                chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
                chart_df = chart_df.dropna().set_index("timestamp")
                if not chart_df.empty:
                    st.line_chart(chart_df)
        with right:
            if {"estimated_cost_for_10mg_usd", "synthetic_feasibility_score"}.issubset(molecules_df.columns):
                summary_df = molecules_df[["estimated_cost_for_10mg_usd", "synthetic_feasibility_score"]].describe().T.round(2)
                st.dataframe(summary_df, use_container_width=True)


def _render_ranking_tab(st, molecules_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Clasament molecule")
    st.caption("Tabel simplificat pentru triere rapida in laborator.")
    filtered = _filter_molecule_frame(st, molecules_df, key_prefix="ranking")
    view = filtered[
        [
            column
            for column in [
                "rank",
                "stare_afisare",
                "smiles_scurt",
                "transformare_afisare",
                "predicted_pIC50",
                "QED",
                "synthetic_feasibility_score",
                "risc_proxy",
                "round",
            ]
            if column in filtered.columns
        ]
    ].rename(
        columns={
            "rank": "Rang",
            "stare_afisare": "Stare",
            "smiles_scurt": "SMILES",
            "transformare_afisare": "Transformare",
            "predicted_pIC50": "pIC50 prezis",
            "QED": "QED",
            "synthetic_feasibility_score": "Fezabilitate",
            "risc_proxy": "Risc proxy",
            "round": "Runda",
        }
    )
    for column in ["pIC50 prezis", "QED", "Fezabilitate", "Risc proxy"]:
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    st.dataframe(view, use_container_width=True, hide_index=True)
    return filtered


def _selection_label(row: pd.Series) -> str:
    return " | ".join(
        [
            f"#{int(safe_float(row.get('rank'), 0))}",
            str(row.get("stare_afisare", "Necunoscut")),
            f"pIC50 {safe_float(row.get('predicted_pIC50'), 0.0):.2f}",
            str(row.get("transformare_afisare", "necunoscut")),
        ]
    )


def _render_detail_note(st, row: pd.Series) -> None:
    state_key = str(row.get("live_status", "necunoscut"))
    message = status_note(state_key)
    market_similarity = safe_float(row.get("max_market_similarity"), 0.0)
    st.markdown(
        f"""
        <div class="lab-note lab-note-info">
            <strong>{status_label(state_key)}</strong><br/>
            {message} Similaritate maxima cu piata: {market_similarity:.2f}.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detail_tab(st, molecules_df: pd.DataFrame) -> None:
    st.subheader("Fisa moleculei")
    filtered = _filter_molecule_frame(st, molecules_df, key_prefix="detail")
    if filtered.empty:
        st.info("Nu exista molecule care sa treaca filtrele curente.")
        return

    top_df = filtered.head(30).copy()
    top_df["eticheta_selectie"] = top_df.apply(_selection_label, axis=1)
    selected_label = st.selectbox("Selecteaza o molecula", options=top_df["eticheta_selectie"].tolist())
    selected_row = top_df.loc[top_df["eticheta_selectie"] == selected_label].iloc[0]

    left, right = st.columns([1.0, 1.25])
    with left:
        image = _render_molecule(str(selected_row["smiles"]))
        if image is not None:
            st.image(image, caption="Structura 2D", use_container_width=False)
        else:
            st.warning("Structura nu a putut fi desenata.")

    with right:
        metrics = st.columns(4)
        metrics[0].metric("Stare", str(selected_row.get("stare_afisare", "Necunoscut")))
        metrics[1].metric("Scor live", f"{safe_float(selected_row.get('live_rank_score'), 0.0):.3f}")
        metrics[2].metric("pIC50 prezis", f"{safe_float(selected_row.get('predicted_pIC50'), 0.0):.2f}")
        metrics[3].metric("QED", f"{safe_float(selected_row.get('QED'), 0.0):.2f}")

        metrics = st.columns(4)
        metrics[0].metric("Fezabilitate", f"{safe_float(selected_row.get('synthetic_feasibility_score'), 0.0):.2f}")
        metrics[1].metric("Risc proxy", f"{safe_float(selected_row.get('reward_hacking_risk'), 0.0):.2f}")
        metrics[2].metric("Similaritate piata", f"{safe_float(selected_row.get('max_market_similarity'), 0.0):.2f}")
        metrics[3].metric("Cost 10 mg", f"${safe_float(selected_row.get('estimated_cost_for_10mg_usd'), 0.0):.2f}")

        _render_detail_note(st, selected_row)

    detail_tabs = st.tabs(["Interpretare", "Componente ranking", "Cost si sinteza", "Date complete"])

    with detail_tabs[0]:
        overview = pd.DataFrame(
            [
                ("Molecula parinte", str(selected_row.get("parent_seed", "n/a"))),
                ("Transformare", str(selected_row.get("transformare_afisare", "n/a"))),
                ("Runda", int(safe_float(selected_row.get("round"), 0))),
                ("SMILES complet", str(selected_row.get("smiles", ""))),
            ],
            columns=["Camp", "Valoare"],
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with detail_tabs[1]:
        ranking_details = pd.DataFrame(
            [
                ("Baza ranking", safe_float(selected_row.get("ranking_component_baza"), 0.0)),
                ("Bonus piata", safe_float(selected_row.get("ranking_component_piata"), 0.0)),
                ("Bonus structura", safe_float(selected_row.get("ranking_component_structura"), 0.0)),
                ("Bonus cost", safe_float(selected_row.get("ranking_component_cost"), 0.0)),
                ("Bonus fezabilitate", safe_float(selected_row.get("ranking_component_fezabilitate"), 0.0)),
                ("Bonus certitudine", safe_float(selected_row.get("ranking_component_certitudine"), 0.0)),
                ("Penalizare risc", safe_float(selected_row.get("ranking_penalizare_risc"), 0.0)),
            ],
            columns=["Componenta", "Valoare"],
        )
        ranking_details["Valoare"] = ranking_details["Valoare"].round(3)
        st.dataframe(ranking_details, use_container_width=True, hide_index=True)

    with detail_tabs[2]:
        cost_details = pd.DataFrame(
            [
                ("Ruta estimata", str(selected_row.get("estimated_route_label", "standard"))),
                ("Multiplicator ruta", safe_float(selected_row.get("estimated_route_multiplier"), 0.0)),
                ("Numar pasi", safe_float(selected_row.get("estimated_step_count"), 0.0)),
                ("Randament pe pas", safe_float(selected_row.get("estimated_step_yield"), 0.0)),
                ("Penalizare randament", safe_float(selected_row.get("estimated_yield_penalty"), 0.0)),
                ("Indice raritate", safe_float(selected_row.get("estimated_rarity_index"), 0.0)),
                ("Complexitate purificare", safe_float(selected_row.get("estimated_purification_complexity"), 0.0)),
                ("Ore laborator", safe_float(selected_row.get("estimated_labor_hours"), 0.0)),
                ("Cost / mmol", safe_float(selected_row.get("estimated_cost_usd_per_mmol"), 0.0)),
                ("Cost 100 mg", safe_float(selected_row.get("estimated_cost_for_100mg_usd"), 0.0)),
                ("Scor cost", safe_float(selected_row.get("estimated_cost_score"), 0.0)),
            ],
            columns=["Indicator", "Valoare"],
        )
        cost_details["Valoare"] = cost_details["Valoare"].round(3)
        st.dataframe(cost_details, use_container_width=True, hide_index=True)

    with detail_tabs[3]:
        st.json({key: value for key, value in selected_row.items() if pd.notna(value)})


def _render_activity_tab(st, rounds_df: pd.DataFrame, log_text: str, source_urls: list[str]) -> None:
    st.subheader("Activitate si trasabilitate")
    left, right = st.columns(2)
    with left:
        st.markdown("**Evenimente recente**")
        if not rounds_df.empty:
            recent = rounds_df.sort_values("timestamp", ascending=False).head(20).copy()
            recent = recent.rename(
                columns={
                    "runda": "Runda",
                    "pas_seed": "Pas",
                    "parinte": "Molecula parinte",
                    "candidati_noi": "Candidati noi",
                    "candidati_promovati": "Promovati",
                    "candidati_totali": "Total",
                    "scor_live_maxim": "Scor live maxim",
                    "timestamp": "Moment",
                }
            )
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("Nu exista inca evenimente de afisat.")

    with right:
        st.markdown("**Jurnal worker**")
        if log_text:
            st.code(log_text, language="text")
        else:
            st.info("Fisierul `worker.log` nu exista inca.")

    with st.expander("Formula de cost si surse", expanded=False):
        st.markdown(build_cost_model_markdown())
        for url in source_urls:
            st.markdown(f"- [{url}]({url})")


def main() -> None:
    st = _load_streamlit()
    st.set_page_config(
        page_title="Statie de lucru pentru laborator",
        page_icon="L",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_theme(st)

    with st.sidebar:
        st.header("Configurare sesiune")
        session_dir = Path(st.text_input("Director sesiune", str(SESSION_DIR)))
        mode = st.selectbox("Mod de lucru", options=list(MODE_LABELS.keys()), format_func=lambda key: MODE_LABELS.get(key, key))
        st.caption(mode_note(mode))
        seed_count = st.slider("Numar seminte", 3, 20, 8, 1)
        rounds = st.slider("Numar runde", 1, 8, 3, 1)
        variants_per_seed = st.slider("Variante per samanta", 5, 120, 30, 5)
        beam_width = st.slider("Latime fascicul", 3, 20, 8, 1)
        refresh_seconds = st.slider("Actualizare automata (secunde)", 2, 20, 4, 1)

    status = safe_json(session_dir / "status.json")

    with st.sidebar:
        st.divider()
        st.header("Actiuni")
        if st.button("Porneste generarea", type="primary", use_container_width=True):
            _launch_worker(
                session_dir=session_dir,
                mode=mode,
                seed_count=seed_count,
                rounds=rounds,
                variants_per_seed=variants_per_seed,
                beam_width=beam_width,
            )
            st.success("Worker-ul a fost pornit.")
            st.rerun()

        if st.button("Opreste generarea", use_container_width=True):
            _terminate_worker(int(status.get("pid", 0) or 0))
            st.warning("Am trimis comanda de oprire.")

        if st.button("Reseteaza sesiunea", use_container_width=True):
            _reset_session_dir(session_dir)
            st.success("Sesiunea a fost resetata.")
            st.rerun()

        if st.button("Actualizeaza acum", use_container_width=True):
            st.rerun()

        st.divider()
        st.header("Ajutor rapid")
        st.markdown("\n".join([f"- {line}" for line in HELP_LINES]))

    render_header_banner(st, f"Mod selectat: {MODE_LABELS.get(mode, mode)}", mode_note(mode))

    @st.fragment(run_every=f"{refresh_seconds}s")
    def _live_panel() -> None:
        live_status = safe_json(session_dir / "status.json")
        molecules_df = prepare_molecule_frame(safe_csv(session_dir / "molecule_generate.csv"))
        rounds_df = safe_csv(session_dir / "rezumat_runde.csv")
        log_text = tail_text(session_dir / "worker.log")
        summary = summarize_molecules(molecules_df)

        render_status_banner(
            st,
            status_label(live_status.get("status", "necunoscut")),
            str(live_status.get("mod_label", MODE_LABELS.get(mode, mode))),
            str(live_status.get("updated_at", "-")),
            str(live_status.get("mesaj", "Nu exista rulare activa.")),
            _running_status(live_status),
        )
        _render_metric_row(st, summary)
        st.progress(_progress_fraction(live_status), text="Progres sesiune")
        render_info_strip(st, session_dir)

        if molecules_df.empty:
            st.warning("Nu exista inca molecule in sesiunea curenta. Configureaza sesiunea si porneste generarea.")
            if log_text:
                st.subheader("Jurnal worker")
                st.code(log_text, language="text")
            return

        tabs = st.tabs(TAB_NAMES)
        with tabs[0]:
            _render_general_tab(st, molecules_df, rounds_df)
        with tabs[1]:
            _render_ranking_tab(st, molecules_df)
        with tabs[2]:
            _render_detail_tab(st, molecules_df)
        with tabs[3]:
            _render_activity_tab(st, rounds_df, log_text, list(live_status.get("source_urls", LITERATURE_SOURCE_URLS)))

    _live_panel()


if __name__ == "__main__":
    main()
