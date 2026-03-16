from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Crippen, rdMolAlign, rdMolDescriptors, rdShapeHelpers

from src.config import PROJECT_ROOT
from src.structure.vina_docking import VinaDockingRescorer
from src.utils.similarity import mol_from_smiles


@dataclass(frozen=True)
class ReferencePose:
    name: str
    smiles: str
    mol_3d: Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@lru_cache(maxsize=4096)
def _embed_3d_smiles(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.25

    with rdBase.BlockLogs():
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            return None

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    return mol


def _copy_embedded_mol(smiles: str):
    embedded = _embed_3d_smiles(smiles)
    if embedded is None:
        return None
    return Chem.Mol(embedded)


def _alignment_score_norm(raw_alignment_score: float) -> float:
    return _clip01(1.0 - math.exp(-max(0.0, raw_alignment_score) / 60.0))


def _pairwise_reference_score(query_smiles: str, reference: ReferencePose) -> dict[str, Any] | None:
    probe = _copy_embedded_mol(query_smiles)
    if probe is None:
        return None

    ref = Chem.Mol(reference.mol_3d)
    with rdBase.BlockLogs():
        probe_contribs = Crippen._GetAtomContribs(probe)
        ref_contribs = Crippen._GetAtomContribs(ref)
        o3a = rdMolAlign.GetCrippenO3A(probe, ref, probe_contribs, ref_contribs)
        raw_alignment = float(o3a.Score())
        o3a.Align()

    shape_similarity = 1.0 - float(rdShapeHelpers.ShapeTanimotoDist(probe, ref))
    protrude_similarity = 1.0 - float(rdShapeHelpers.ShapeProtrudeDist(probe, ref))
    usr_similarity = float(
        rdMolDescriptors.GetUSRScore(
            rdMolDescriptors.GetUSR(probe),
            rdMolDescriptors.GetUSR(ref),
        )
    )
    alignment_norm = _alignment_score_norm(raw_alignment)

    docking_rescore = _clip01(
        0.35 * shape_similarity
        + 0.20 * protrude_similarity
        + 0.20 * usr_similarity
        + 0.25 * alignment_norm
    )

    return {
        "closest_pose_reference": reference.name,
        "closest_pose_smiles": reference.smiles,
        "shape_similarity": float(shape_similarity),
        "protrude_similarity": float(protrude_similarity),
        "usr_similarity": float(usr_similarity),
        "alignment_score_raw": raw_alignment,
        "alignment_score_norm": alignment_norm,
        "docking_rescore": docking_rescore,
        "docking_backend": "reference_ligand",
    }


def _load_reference_poses(reference_csv: Path | None = None) -> list[ReferencePose]:
    path = reference_csv or (PROJECT_ROOT / "data" / "processed" / "marketed_egfr_benchmark.csv")
    if not path.exists():
        return []

    df = pd.read_csv(path, low_memory=False)
    if "smiles" not in df.columns:
        return []

    refs: list[ReferencePose] = []
    for _, row in df.iterrows():
        smiles = row.get("smiles")
        name = str(row.get("name", "reference"))
        mol_3d = _copy_embedded_mol(str(smiles))
        if mol_3d is None:
            continue
        refs.append(ReferencePose(name=name, smiles=str(smiles), mol_3d=mol_3d))
    return refs


class ReferenceLigandRescorer:
    def __init__(self, reference_csv: Path | None = None):
        self.reference_poses = _load_reference_poses(reference_csv)

    def is_available(self) -> bool:
        return bool(self.reference_poses)

    def score_smiles(self, smiles: str) -> dict[str, Any]:
        if not self.reference_poses:
            return {
                "closest_pose_reference": None,
                "closest_pose_smiles": None,
                "shape_similarity": 0.0,
                "protrude_similarity": 0.0,
                "usr_similarity": 0.0,
                "alignment_score_raw": 0.0,
                "alignment_score_norm": 0.0,
                "docking_rescore": 0.0,
                "docking_backend": "unavailable",
            }

        best: dict[str, Any] | None = None
        for reference in self.reference_poses:
            scored = _pairwise_reference_score(smiles, reference)
            if scored is None:
                continue
            if best is None or float(scored["docking_rescore"]) > float(best["docking_rescore"]):
                best = scored

        if best is None:
            return {
                "closest_pose_reference": None,
                "closest_pose_smiles": None,
                "shape_similarity": 0.0,
                "protrude_similarity": 0.0,
                "usr_similarity": 0.0,
                "alignment_score_raw": 0.0,
                "alignment_score_norm": 0.0,
                "docking_rescore": 0.0,
                "docking_backend": "failed",
            }
        return best


class StructuralConsensusRescorer:
    def __init__(
        self,
        *,
        backend: str = "auto",
        pose_dir: Path | None = None,
        vina_cpu: int = 1,
        vina_exhaustiveness: int = 6,
        vina_num_modes: int = 5,
    ):
        normalized_backend = backend.lower().strip()
        if normalized_backend not in {"auto", "reference", "vina"}:
            raise ValueError("backend must be one of: auto, reference, vina")

        self.backend = normalized_backend
        self.reference_rescorer = ReferenceLigandRescorer() if normalized_backend in {"auto", "reference"} else None
        self.vina_rescorer = (
            VinaDockingRescorer(
                pose_dir=pose_dir,
                cpu=vina_cpu,
                exhaustiveness=vina_exhaustiveness,
                num_modes=vina_num_modes,
            )
            if normalized_backend in {"auto", "vina"}
            else None
        )

    def is_available(self) -> bool:
        reference_available = self.reference_rescorer.is_available() if self.reference_rescorer is not None else False
        vina_available = self.vina_rescorer.is_available() if self.vina_rescorer is not None else False
        return reference_available or vina_available

    def score_smiles(self, smiles: str, ligand_name: str | None = None) -> dict[str, Any]:
        reference_payload: dict[str, Any] = {}
        vina_payload: dict[str, Any] = {}

        if self.reference_rescorer is not None and self.reference_rescorer.is_available():
            reference_payload = self.reference_rescorer.score_smiles(smiles)
        if self.vina_rescorer is not None and self.vina_rescorer.is_available():
            vina_payload = self.vina_rescorer.score_smiles(smiles, ligand_name=ligand_name)

        reference_support = float(reference_payload.get("docking_rescore", 0.0))
        vina_support = float(vina_payload.get("vina_rescore", 0.0))
        vina_ok = vina_payload.get("vina_status") == "ok"
        reference_ok = bool(reference_payload) and reference_payload.get("docking_backend") not in {"failed", "unavailable"}

        out: dict[str, Any] = {
            "closest_pose_reference": reference_payload.get("closest_pose_reference"),
            "closest_pose_smiles": reference_payload.get("closest_pose_smiles"),
            "shape_similarity": float(reference_payload.get("shape_similarity", 0.0)),
            "protrude_similarity": float(reference_payload.get("protrude_similarity", 0.0)),
            "usr_similarity": float(reference_payload.get("usr_similarity", 0.0)),
            "alignment_score_raw": float(reference_payload.get("alignment_score_raw", 0.0)),
            "alignment_score_norm": float(reference_payload.get("alignment_score_norm", 0.0)),
            "reference_docking_rescore": reference_support,
            "reference_backend": reference_payload.get("docking_backend", "unavailable"),
            "vina_affinity_kcal": vina_payload.get("vina_affinity_kcal"),
            "vina_best_mode": vina_payload.get("vina_best_mode"),
            "vina_best_rmsd_lb": vina_payload.get("vina_best_rmsd_lb"),
            "vina_best_rmsd_ub": vina_payload.get("vina_best_rmsd_ub"),
            "vina_pose_count": int(vina_payload.get("vina_pose_count", 0)),
            "vina_rescore": vina_support,
            "vina_status": vina_payload.get("vina_status", "unavailable"),
            "docking_pose_path": vina_payload.get("docking_pose_path"),
        }

        if vina_ok and reference_ok:
            out["docking_rescore"] = 0.65 * vina_support + 0.35 * reference_support
            out["docking_backend"] = "consensus_vina_reference"
        elif vina_ok:
            out["docking_rescore"] = vina_support
            out["docking_backend"] = "autodock_vina"
        elif reference_ok:
            out["docking_rescore"] = reference_support
            out["docking_backend"] = reference_payload.get("docking_backend", "reference_ligand")
        else:
            out["docking_rescore"] = 0.0
            out["docking_backend"] = "unavailable"

        return out
