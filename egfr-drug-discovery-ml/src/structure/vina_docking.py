from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from src.config import PROJECT_ROOT
from src.utils.similarity import mol_from_smiles


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_token(value: str, fallback: str = "ligand") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:96] or fallback


def _embed_smiles_3d(smiles: str):
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


def _affinity_to_support(affinity_kcal: float | None) -> float:
    if affinity_kcal is None or math.isnan(affinity_kcal):
        return 0.0
    return _clip01(((-affinity_kcal) - 5.0) / 5.0)


@dataclass(frozen=True)
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    @classmethod
    def from_file(cls, path: Path) -> "DockingBox":
        values: dict[str, float] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw_line:
                continue
            key, raw_value = [part.strip() for part in raw_line.split("=", 1)]
            values[key] = float(raw_value)
        required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"Docking box file {path} is missing values: {missing}")
        return cls(**{key: values[key] for key in required})


@dataclass(frozen=True)
class VinaDockingConfig:
    vina_executable: Path
    prepare_ligand_executable: Path
    receptor_pdbqt: Path
    docking_box: DockingBox
    scoring: str = "vina"
    exhaustiveness: int = 6
    num_modes: int = 5
    cpu: int = 1
    seed: int = 42
    timeout_seconds: int = 240


def _resolve_prepare_ligand_executable() -> Path | None:
    local_candidates = [
        Path(sys.executable).with_name("mk_prepare_ligand.exe"),
        Path(sys.executable).with_name("mk_prepare_ligand"),
        PROJECT_ROOT / ".venv" / "Scripts" / "mk_prepare_ligand.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    resolved = shutil.which("mk_prepare_ligand.exe") or shutil.which("mk_prepare_ligand")
    return Path(resolved) if resolved else None


def build_default_vina_config(
    *,
    exhaustiveness: int = 6,
    cpu: int = 1,
    num_modes: int = 5,
    timeout_seconds: int = 240,
) -> VinaDockingConfig | None:
    vina_executable = PROJECT_ROOT / "tools" / "vina" / "vina.exe"
    receptor_pdbqt = PROJECT_ROOT / "data" / "external" / "egfr_receptor" / "prepared" / "4WKQ_egfr.pdbqt"
    box_file = PROJECT_ROOT / "data" / "external" / "egfr_receptor" / "prepared" / "4WKQ_egfr.box.txt"
    prepare_ligand = _resolve_prepare_ligand_executable()

    if not vina_executable.exists() or not receptor_pdbqt.exists() or not box_file.exists() or prepare_ligand is None:
        return None

    return VinaDockingConfig(
        vina_executable=vina_executable,
        prepare_ligand_executable=prepare_ligand,
        receptor_pdbqt=receptor_pdbqt,
        docking_box=DockingBox.from_file(box_file),
        exhaustiveness=max(1, int(exhaustiveness)),
        cpu=max(1, int(cpu)),
        num_modes=max(1, int(num_modes)),
        timeout_seconds=max(30, int(timeout_seconds)),
    )


def _parse_vina_stdout(stdout: str) -> dict[str, Any]:
    modes: list[dict[str, float | int]] = []
    in_table = False

    for line in stdout.splitlines():
        if "mode |   affinity | dist from best mode" in line:
            in_table = True
            continue
        if not in_table:
            continue
        match = re.match(
            r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$",
            line,
        )
        if match:
            modes.append(
                {
                    "mode": int(match.group(1)),
                    "affinity_kcal": float(match.group(2)),
                    "rmsd_lb": float(match.group(3)),
                    "rmsd_ub": float(match.group(4)),
                }
            )

    best = modes[0] if modes else None
    best_affinity = float(best["affinity_kcal"]) if best else math.nan
    return {
        "vina_pose_count": len(modes),
        "vina_best_mode": int(best["mode"]) if best else None,
        "vina_affinity_kcal": best_affinity,
        "vina_best_rmsd_lb": float(best["rmsd_lb"]) if best else math.nan,
        "vina_best_rmsd_ub": float(best["rmsd_ub"]) if best else math.nan,
        "vina_rescore": _affinity_to_support(best_affinity if best else None),
    }


class VinaDockingRescorer:
    def __init__(
        self,
        *,
        config: VinaDockingConfig | None = None,
        pose_dir: Path | None = None,
        exhaustiveness: int = 6,
        cpu: int = 1,
        num_modes: int = 5,
        timeout_seconds: int = 240,
    ):
        self.config = config or build_default_vina_config(
            exhaustiveness=exhaustiveness,
            cpu=cpu,
            num_modes=num_modes,
            timeout_seconds=timeout_seconds,
        )
        self.pose_dir = pose_dir
        if self.pose_dir is not None:
            self.pose_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        return self.config is not None

    def _default_payload(self, status: str) -> dict[str, Any]:
        return {
            "vina_pose_count": 0,
            "vina_best_mode": None,
            "vina_affinity_kcal": math.nan,
            "vina_best_rmsd_lb": math.nan,
            "vina_best_rmsd_ub": math.nan,
            "vina_rescore": 0.0,
            "vina_status": status,
            "docking_pose_path": None,
        }

    def _write_input_sdf(self, smiles: str, out_path: Path) -> bool:
        mol = _embed_smiles_3d(smiles)
        if mol is None:
            return False
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()
        return True

    def score_smiles(self, smiles: str, ligand_name: str | None = None) -> dict[str, Any]:
        if self.config is None:
            return self._default_payload("unavailable")

        token = _safe_token(ligand_name or smiles[:48], fallback="ligand")
        with tempfile.TemporaryDirectory(prefix="oncoforge_vina_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            sdf_path = tmp_dir / f"{token}.sdf"
            ligand_pdbqt = tmp_dir / f"{token}.pdbqt"
            docked_pdbqt = tmp_dir / f"{token}_out.pdbqt"

            if not self._write_input_sdf(smiles, sdf_path):
                return self._default_payload("embed_failed")

            prep_command = [
                str(self.config.prepare_ligand_executable),
                "-i",
                str(sdf_path),
                "-o",
                str(ligand_pdbqt),
            ]
            try:
                subprocess.run(
                    prep_command,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.config.timeout_seconds,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return self._default_payload("ligand_prep_failed")

            dock_command = [
                str(self.config.vina_executable),
                "--receptor",
                str(self.config.receptor_pdbqt),
                "--ligand",
                str(ligand_pdbqt),
                "--center_x",
                str(self.config.docking_box.center_x),
                "--center_y",
                str(self.config.docking_box.center_y),
                "--center_z",
                str(self.config.docking_box.center_z),
                "--size_x",
                str(self.config.docking_box.size_x),
                "--size_y",
                str(self.config.docking_box.size_y),
                "--size_z",
                str(self.config.docking_box.size_z),
                "--scoring",
                self.config.scoring,
                "--cpu",
                str(self.config.cpu),
                "--seed",
                str(self.config.seed),
                "--exhaustiveness",
                str(self.config.exhaustiveness),
                "--num_modes",
                str(self.config.num_modes),
                "--out",
                str(docked_pdbqt),
            ]
            try:
                completed = subprocess.run(
                    dock_command,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return self._default_payload("docking_timeout")
            except subprocess.CalledProcessError:
                return self._default_payload("docking_failed")

            parsed = _parse_vina_stdout(completed.stdout)
            pose_path_str = None
            if self.pose_dir is not None and docked_pdbqt.exists():
                saved_pose = self.pose_dir / f"{token}_docked.pdbqt"
                saved_pose.write_bytes(docked_pdbqt.read_bytes())
                pose_path_str = str(saved_pose)

            parsed["vina_status"] = "ok" if parsed["vina_pose_count"] > 0 else "no_pose"
            parsed["docking_pose_path"] = pose_path_str
            return parsed
