from __future__ import annotations

import unittest

import pandas as pd

from src.agents.external_evidence_agent import add_external_evidence_agent_ranking
from src.agents.multi_agent import add_structure_agent_ranking, build_default_scorer, score_smiles_list
from src.config import PROJECT_ROOT
from src.data.fetch_pubchem_egfr import ensure_pubchem_reference
from src.evaluation.cross_database_validation import CrossDatabaseValidator
from src.feasibility.experimental_readiness import add_experimental_readiness
from src.generation.analog_generator import generate_string_mutations
from src.generation.medchem_mutations import generate_medchem_outcomes
from src.knowledge import BUZZWORD_ENTRIES
from src.generation.rgroup_generator import generate_rgroup_variants
from src.rl.environment import VerifiableMoleculeEnv
from src.structure import StructuralConsensusRescorer
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

    def test_medchem_generator_applies_hard_constraints_and_scores(self):
        outcomes = generate_medchem_outcomes(
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            max_variants=24,
        )
        self.assertGreater(len(outcomes), 0)
        self.assertTrue(all(outcome.hard_constraint_pass for outcome in outcomes))
        self.assertTrue(all(outcome.generator_priority_score > 0 for outcome in outcomes))
        self.assertTrue(all(outcome.parent_similarity >= 0.28 for outcome in outcomes))

    def test_rl_environment_exposes_unique_action_ids(self):
        env = VerifiableMoleculeEnv(
            ["COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"],
            scorer=self.scorer,
            max_steps=2,
            max_variants_per_step=24,
            max_actions_per_family=2,
            max_actions_total=10,
        )
        env.reset()
        actions = env.available_actions()
        self.assertGreater(len(actions), 0)
        self.assertEqual(len({action.action_id for action in actions}), len(actions))
        self.assertTrue(all(action.selection_rank >= 1 for action in actions))

    def test_structural_consensus_rescorer_returns_structural_fields(self):
        rescorer = StructuralConsensusRescorer(backend="reference")
        result = rescorer.score_smiles("OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1", ligand_name="smoke_test")
        for key in ["docking_rescore", "reference_docking_rescore", "docking_backend"]:
            self.assertIn(key, result)

    def test_buzzword_glossary_has_core_entries(self):
        self.assertGreaterEqual(len(BUZZWORD_ENTRIES), 25)
        terms = {entry.term for entry in BUZZWORD_ENTRIES}
        for expected in ["Multi-agent system", "Verifiable reward", "EGFR", "Docking"]:
            self.assertIn(expected, terms)

    def test_structure_agent_ranking_adds_structural_fields(self):
        df = pd.DataFrame(
            [
                {
                    "smiles": "OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
                    "predicted_pIC50": 8.9,
                    "final_score": 10.2,
                    "docking_rescore": 0.71,
                    "interaction_support_score": 0.66,
                    "interaction_key_residue_count": 3,
                    "feasibility_score": 0.82,
                    "reward_hacking_risk": 0.12,
                    "audit_status": "pass",
                    "veto": False,
                    "uncertainty": 0.12,
                },
                {
                    "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
                    "predicted_pIC50": 8.7,
                    "final_score": 9.8,
                    "docking_rescore": 0.59,
                    "interaction_support_score": 0.48,
                    "interaction_key_residue_count": 2,
                    "feasibility_score": 0.74,
                    "reward_hacking_risk": 0.21,
                    "audit_status": "pass",
                    "veto": False,
                    "uncertainty": 0.16,
                },
            ]
        )
        ranked = add_structure_agent_ranking(df)
        for key in ["structure_agent_support", "structure_augmented_score", "structure_agent_status", "structure_rank"]:
            self.assertIn(key, ranked.columns)

    def test_experimental_readiness_adds_readiness_fields(self):
        df = pd.DataFrame(
            [
                {
                    "smiles": "OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
                    "predicted_pIC50": 8.9,
                    "final_score": 10.2,
                    "docking_rescore": 0.71,
                    "interaction_support_score": 0.66,
                    "interaction_key_residue_count": 3,
                    "feasibility_score": 0.82,
                    "feasibility_status": "pass",
                    "max_active_similarity": 0.61,
                    "source_support_score": 0.45,
                    "traceability_score": 1.0,
                    "synthetic_ease_score": 0.63,
                    "QED": 0.58,
                    "reward_hacking_risk": 0.12,
                    "audit_status": "pass",
                    "veto": False,
                    "uncertainty": 0.12,
                }
            ]
        )
        ready = add_experimental_readiness(df)
        for key in ["experimental_readiness_score", "experimental_readiness_status", "experimental_track", "experimental_readiness_priority"]:
            self.assertIn(key, ready.columns)

    def test_cross_database_validator_adds_support_fields(self):
        validator = CrossDatabaseValidator()
        df = pd.DataFrame(
            [
                {
                    "smiles": "OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
                    "predicted_pIC50": 8.9,
                    "final_score": 10.2,
                }
            ]
        )
        validated = validator.validate_frame(df)
        for key in [
            "cross_database_consensus_score",
            "cross_database_independent_support_count",
            "cross_database_status",
            "cross_database_priority",
        ]:
            self.assertIn(key, validated.columns)

    def test_pubchem_reference_has_enriched_columns(self):
        path = ensure_pubchem_reference()
        df = pd.read_csv(path, nrows=5)
        for key in [
            "pubchem_enriched_evidence_score",
            "pubchem_relevance_score",
            "pubchem_orthogonal_support_score",
            "virtual_proxy_fraction",
        ]:
            self.assertIn(key, df.columns)

    def test_external_evidence_agent_adds_fields(self):
        df = pd.DataFrame(
            [
                {
                    "smiles": "OCCNc1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
                    "predicted_pIC50": 8.9,
                    "final_score": 10.2,
                    "cross_database_consensus_score": 0.62,
                    "cross_database_independent_support_count": 3,
                    "cross_database_external_support_count": 2,
                    "cross_database_status": "strong",
                    "pubchem_support_score": 0.71,
                    "bindingdb_support_score": 0.63,
                    "iuphar_support_score": 0.58,
                    "source_support_score": 0.42,
                    "max_active_similarity": 0.66,
                    "reward_hacking_risk": 0.12,
                    "veto": False,
                }
            ]
        )
        enriched = add_external_evidence_agent_ranking(df)
        for key in [
            "external_evidence_support",
            "external_evidence_guardrail",
            "external_evidence_status",
            "external_evidence_priority",
        ]:
            self.assertIn(key, enriched.columns)


if __name__ == "__main__":
    unittest.main()
