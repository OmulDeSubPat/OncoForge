from __future__ import annotations

import unittest

import pandas as pd

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.generation.analog_generator import generate_string_mutations
from src.generation.rgroup_generator import generate_rgroup_variants
from src.utils.similarity import mol_from_smiles


class OncoForgeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = build_default_scorer()

    def test_marketed_benchmark_smiles_are_valid(self):
        benchmark_path = PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv"
        df = pd.read_csv(benchmark_path)
        invalid = [row["name"] for _, row in df.iterrows() if mol_from_smiles(row["smiles"]) is None]
        self.assertEqual(invalid, [], f"Invalid benchmark SMILES found: {invalid}")

    def test_multi_agent_scorer_returns_expected_fields(self):
        result = self.scorer.score("OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1")
        for key in [
            "predicted_pIC50",
            "QED",
            "reward_hacking_risk",
            "naive_reward",
            "anti_hacking_penalty",
            "verified_reward",
            "audit_status",
            "veto",
        ]:
            self.assertIn(key, result)

    def test_ranking_outputs_audit_diagnostics(self):
        df = score_smiles_list(
            [
                "OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
                "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
                "CN(C)CCOC(c1ccccc1)c1ccccc1",
            ],
            scorer=self.scorer,
        )
        for key in ["naive_score", "naive_rank", "audit_demote_positions", "selection_bucket"]:
            self.assertIn(key, df.columns)

    def test_string_mutation_generator_produces_valid_variants(self):
        variants = generate_string_mutations("c1ccccc1Cl", max_variants=20)
        self.assertGreater(len(variants), 0)
        self.assertTrue(all(mol_from_smiles(smiles) is not None for smiles in variants))

    def test_rgroup_generator_produces_valid_variants(self):
        variants = generate_rgroup_variants("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", max_variants=20)
        self.assertGreater(len(variants), 0)
        self.assertTrue(all(mol_from_smiles(smiles) is not None for smiles in variants))


if __name__ == "__main__":
    unittest.main()
