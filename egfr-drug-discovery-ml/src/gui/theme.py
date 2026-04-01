from __future__ import annotations

from pathlib import Path
from typing import Any


LAB_THEME = {
    "bg": "#f6faf9",
    "panel": "#ffffff",
    "panel_soft": "#edf4f2",
    "border": "#c8d8d4",
    "text": "#102a2a",
    "muted": "#4d6360",
    "accent": "#0f766e",
    "accent_dark": "#0b5f59",
    "accent_soft": "#d8f0ed",
    "warning": "#9a3412",
    "warning_soft": "#ffedd5",
    "danger": "#b91c1c",
    "danger_soft": "#fee2e2",
    "success": "#166534",
    "success_soft": "#dcfce7",
    "info": "#1d4ed8",
    "info_soft": "#dbeafe",
    "shadow": "0 14px 38px rgba(16, 42, 42, 0.08)",
}


LAB_LAYOUT = {
    "hero_border_radius": "22px",
    "card_border_radius": "18px",
    "chip_border_radius": "999px",
    "content_max_width": "1320px",
}


def _css_block() -> str:
    return f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(216, 240, 237, 0.95), transparent 28%),
            radial-gradient(circle at top right, rgba(219, 234, 254, 0.78), transparent 25%),
            linear-gradient(180deg, {LAB_THEME["bg"]} 0%, #eef4f2 100%);
        color: {LAB_THEME["text"]};
    }}

    [data-testid="stAppViewContainer"] {{
        max-width: {LAB_LAYOUT["content_max_width"]};
        margin: 0 auto;
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    .lab-hero,
    .lab-panel,
    .lab-status,
    .lab-strip,
    .lab-note {{
        border: 1px solid {LAB_THEME["border"]};
        border-radius: {LAB_LAYOUT["card_border_radius"]};
        background: linear-gradient(180deg, {LAB_THEME["panel"]} 0%, {LAB_THEME["panel_soft"]} 100%);
        box-shadow: {LAB_THEME["shadow"]};
    }}

    .lab-hero {{
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }}

    .lab-hero h1 {{
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.2;
        color: {LAB_THEME["text"]};
    }}

    .lab-hero p {{
        margin: 0.45rem 0 0 0;
        color: {LAB_THEME["muted"]};
        line-height: 1.5;
    }}

    .lab-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: {LAB_LAYOUT["chip_border_radius"]};
        padding: 0.26rem 0.7rem;
        background: {LAB_THEME["accent_soft"]};
        color: {LAB_THEME["accent_dark"]};
        border: 1px solid rgba(15, 118, 110, 0.18);
        font-size: 0.88rem;
        font-weight: 700;
    }}

    .lab-status {{
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
    }}

    .lab-status-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }}

    .lab-stat {{
        border: 1px solid {LAB_THEME["border"]};
        border-radius: 16px;
        padding: 0.75rem 0.85rem;
        background: rgba(255, 255, 255, 0.85);
    }}

    .lab-stat strong {{
        display: block;
        font-size: 0.86rem;
        color: {LAB_THEME["muted"]};
        margin-bottom: 0.18rem;
    }}

    .lab-stat span {{
        font-size: 1.08rem;
        font-weight: 800;
        color: {LAB_THEME["text"]};
    }}

    .lab-strip {{
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }}

    .lab-strip small {{
        color: {LAB_THEME["muted"]};
        display: block;
        margin-top: 0.15rem;
    }}

    .lab-note {{
        padding: 0.9rem 1rem;
        border-left: 5px solid {LAB_THEME["accent"]};
        color: {LAB_THEME["text"]};
    }}

    .lab-note-warning {{
        border-left-color: {LAB_THEME["warning"]};
        background: linear-gradient(180deg, #fffdf9 0%, {LAB_THEME["warning_soft"]} 100%);
    }}

    .lab-note-danger {{
        border-left-color: {LAB_THEME["danger"]};
        background: linear-gradient(180deg, #fffafa 0%, {LAB_THEME["danger_soft"]} 100%);
    }}

    .lab-note-success {{
        border-left-color: {LAB_THEME["success"]};
        background: linear-gradient(180deg, #fcfffd 0%, {LAB_THEME["success_soft"]} 100%);
    }}

    .lab-note-info {{
        border-left-color: {LAB_THEME["info"]};
        background: linear-gradient(180deg, #fbfdff 0%, {LAB_THEME["info_soft"]} 100%);
    }}

    .lab-kpi {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }}

    .lab-kpi-card {{
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        border: 1px solid {LAB_THEME["border"]};
        background: rgba(255,255,255,0.95);
    }}

    .lab-kpi-card .label {{
        display: block;
        color: {LAB_THEME["muted"]};
        font-size: 0.88rem;
        margin-bottom: 0.15rem;
        font-weight: 700;
    }}

    .lab-kpi-card .value {{
        color: {LAB_THEME["text"]};
        font-size: 1.12rem;
        font-weight: 800;
    }}

    .lab-progress {
        height: 0.75rem;
        border-radius: 999px;
    }

    @media (max-width: 980px) {{
        .lab-status-grid,
        .lab-kpi {{
            grid-template-columns: 1fr 1fr;
        }}
        [data-testid="stAppViewContainer"] {{
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }}
    }}

    @media (max-width: 640px) {{
        .lab-status-grid,
        .lab-kpi {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """


def inject_global_theme(st: Any) -> None:
    """Injecteaza o tema globala cu contrast ridicat pentru Streamlit."""
    st.markdown(_css_block(), unsafe_allow_html=True)


def render_header_banner(st: Any, mode_label: str, mode_note: str) -> None:
    """Renderizeaza un banner de inceput clar si usor de citit."""
    st.markdown(
        f"""
        <div class="lab-hero">
            <div class="lab-chip">Flux live de laborator</div>
            <h1>Monitorizare si triere molecule generate</h1>
            <p>{mode_label}</p>
            <p>{mode_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(
    st: Any,
    status_label: str,
    mode_label: str,
    updated_at: str,
    message: str,
    running: bool,
) -> None:
    """Arata starea curenta a sesiunii intr-un bloc foarte vizibil."""
    dot_class = "lab-chip" if running else "lab-chip"
    chip_text = "In rulare" if running else "Oprit"
    st.markdown(
        f"""
        <div class="lab-status">
            <div class="lab-strip">
                <div>
                    <div class="{dot_class}">{chip_text}</div>
                    <small>{status_label} | {mode_label} | {updated_at}</small>
                </div>
                <div style="max-width: 760px; color: {LAB_THEME["text"]}; font-weight: 600;">
                    {message}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_strip(st: Any, session_dir: str | Path) -> None:
    """Arata informatii de baza despre sesiunea curenta."""
    st.markdown(
        f"""
        <div class="lab-strip">
            <div>
                <strong>Folder sesiune</strong>
                <small>{session_dir}</small>
            </div>
            <div>
                <strong>Regula de lucru</strong>
                <small>Mai intai monitorizare, apoi selectie, apoi detaliu molecula.</small>
            </div>
            <div>
                <strong>Mod de prezentare</strong>
                <small>Contrast ridicat, aliniat pentru laborator si triere rapida.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "LAB_LAYOUT",
    "LAB_THEME",
    "inject_global_theme",
    "render_header_banner",
    "render_info_strip",
    "render_status_banner",
]
