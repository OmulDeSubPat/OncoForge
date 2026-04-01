from __future__ import annotations

from typing import Any


STATUS_LABELS = {
    "pornit": "Pornit",
    "in_rulare": "In rulare",
    "finalizat": "Finalizat",
    "eroare": "Eroare",
    "necunoscut": "Necunoscut",
    "promovata": "Promitatoare",
    "revizie": "Necesita revizie",
    "respinsa": "Respinsa",
}

STATUS_NOTES = {
    "promovata": "Molecula poate intra in shortlist-ul de lucru.",
    "revizie": "Molecula merita verificare suplimentara inainte de promovare.",
    "respinsa": "Molecula nu este prioritara in forma actuala.",
}

MODE_NOTES = {
    "explorare": "Bun pentru scanare rapida a spatiului chimic.",
    "ghidat_ai": "Potrivit cand vrei variante curate in jurul unor seminte puternice.",
    "iterativ": "Cea mai buna alegere pentru optimizare progresiva pe mai multe runde.",
}

HELP_LINES = [
    "Configurezi sesiunea din bara laterala.",
    "Pornesti generarea cu butonul principal.",
    "Urmaresti mai intai tabul Panou general.",
    "Alegi moleculele din Clasament si le verifici in Fisa moleculei.",
]

TAB_NAMES = [
    "Panou general",
    "Clasament",
    "Fisa moleculei",
    "Activitate",
]


def status_label(value: Any) -> str:
    key = str(value or "necunoscut").strip().lower()
    return STATUS_LABELS.get(key, str(value or "Necunoscut"))


def status_note(value: Any) -> str:
    key = str(value or "necunoscut").strip().lower()
    return STATUS_NOTES.get(key, "Molecula poate fi evaluata din fisa de mai jos.")


def mode_note(value: Any) -> str:
    key = str(value or "").strip().lower()
    return MODE_NOTES.get(key, "")
