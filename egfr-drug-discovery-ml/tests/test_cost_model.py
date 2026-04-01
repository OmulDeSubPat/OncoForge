from __future__ import annotations

import pandas as pd

from src.economics.cost_model import add_cost_estimates, build_cost_model_markdown, estimate_molecule_cost


def test_estimate_molecule_cost_returns_positive_outputs():
    result = estimate_molecule_cost(
        {
            "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)C=C",
            "SA_score": 3.4,
            "MW": 386.8,
            "LogP": 3.7,
            "ring_count": 4,
            "rotatable_bonds": 6,
            "synthetic_route": "snar_diversification",
            "reaction_family": "snar",
            "top5_train_similarity": 0.62,
            "max_train_similarity": 0.78,
            "synthetic_feasibility_score": 0.80,
            "medchem_realism_score": 0.77,
            "transformation_confidence_score": 0.74,
            "alert_count": 0,
            "severe_alert_count": 0,
        }
    )

    assert result["estimated_cost_usd_per_mmol"] > 0
    assert result["estimated_cost_for_10mg_usd"] > 0
    assert 0 < result["estimated_cost_score"] <= 1
    assert result["estimated_cost_band"] in {"foarte scazut", "scazut", "mediu", "ridicat", "foarte ridicat"}


def test_add_cost_estimates_appends_expected_columns():
    df = pd.DataFrame(
        [
            {
                "smiles": "CCO",
                "SA_score": 2.0,
                "synthetic_feasibility_score": 0.85,
                "medchem_realism_score": 0.80,
                "transformation_confidence_score": 0.75,
            }
        ]
    )

    enriched = add_cost_estimates(df)

    assert "estimated_cost_usd_per_mmol" in enriched.columns
    assert "estimated_cost_for_10mg_usd" in enriched.columns
    assert "estimated_step_count" in enriched.columns


def test_cost_model_markdown_lists_sources():
    note = build_cost_model_markdown()
    assert "CoPriNet" in note
    assert "RouteScore" in note
    assert "http" in note
