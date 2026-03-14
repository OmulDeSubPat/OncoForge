from __future__ import annotations

from src.agents.multi_agent import MultiAgentScorer, build_default_scorer


def load_default_scorer(models=None) -> MultiAgentScorer:
    return build_default_scorer(models=models)


def score_molecule(smiles: str, models=None, scorer: MultiAgentScorer | None = None) -> dict:
    active_scorer = scorer or load_default_scorer(models=models)
    return active_scorer.score(smiles)


def main():
    scorer = load_default_scorer()

    sample_smiles = [
        "CCO",
        "c1ccccc1",
        "CN(C)CCOC(c1ccccc1)c1ccccc1",
    ]

    for smiles in sample_smiles:
        print(score_molecule(smiles, scorer=scorer))


if __name__ == "__main__":
    main()
