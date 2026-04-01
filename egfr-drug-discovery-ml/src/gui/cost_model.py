from __future__ import annotations

from src.economics.cost_model import (
    LITERATURE_SOURCE_URLS,
    add_cost_estimates,
    build_cost_model_markdown,
    estimate_molecule_cost,
)


def attach_cost_estimates(df):
    return add_cost_estimates(df)


__all__ = [
    "LITERATURE_SOURCE_URLS",
    "attach_cost_estimates",
    "build_cost_model_markdown",
    "estimate_molecule_cost",
]
