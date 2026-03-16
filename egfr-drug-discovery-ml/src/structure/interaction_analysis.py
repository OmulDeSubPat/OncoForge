from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT


AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TYR", "TRP", "PRO"}
ACIDIC_RESIDUES = {"ASP", "GLU"}
BASIC_RESIDUES = {"LYS", "ARG", "HIS"}
KEY_RESIDUES = {"LEU718", "VAL726", "ALA743", "LYS745", "MET793", "CYS797", "ASP800", "ASP855"}
HBOND_RELEVANT_ELEMENTS = {"N", "O", "S", "F"}
HYDROPHOBIC_LIGAND_ELEMENTS = {"C", "F", "Cl", "Br", "I", "S"}
ACIDIC_ATOM_NAMES = {"OD1", "OD2", "OE1", "OE2"}
BASIC_ATOM_NAMES = {"NZ", "NH1", "NH2", "NE", "ND1", "NE2"}
AROMATIC_SIDECHAIN_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class ParsedAtom:
    atom_name: str
    residue_name: str
    residue_id: str
    element: str
    coord: np.ndarray
    atom_type: str | None = None


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize_element(raw: str) -> str:
    token = raw.strip().upper()
    if not token:
        return "C"
    if token.startswith("CL"):
        return "Cl"
    if token.startswith("BR"):
        return "Br"
    if token.startswith("NA") or token.startswith("N"):
        return "N"
    if token.startswith("OA") or token.startswith("O"):
        return "O"
    if token.startswith("SA") or token.startswith("S"):
        return "S"
    if token.startswith("HD") or token.startswith("H"):
        return "H"
    if token.startswith("F"):
        return "F"
    if token.startswith("I"):
        return "I"
    if token.startswith("P"):
        return "P"
    return token.capitalize()


def _residue_label(residue_name: str, residue_id: str) -> str:
    return f"{residue_name}{residue_id}"


@lru_cache(maxsize=1)
def _load_receptor_atoms(receptor_pdb: str | None = None) -> tuple[list[ParsedAtom], dict[str, np.ndarray]]:
    receptor_path = Path(receptor_pdb) if receptor_pdb else (PROJECT_ROOT / "data" / "external" / "egfr_receptor" / "4WKQ.pdb")
    atoms: list[ParsedAtom] = []
    aromatic_groups: dict[str, list[np.ndarray]] = defaultdict(list)

    for line in receptor_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21].strip() or "A"
        if chain_id != "A":
            continue
        residue_name = line[17:20].strip().upper()
        residue_id = line[22:26].strip()
        atom_name = line[12:16].strip().upper()
        element = _normalize_element(line[76:78] if len(line) >= 78 else atom_name)
        if element == "H":
            continue
        coord = np.asarray(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ],
            dtype=float,
        )
        residue_label = _residue_label(residue_name, residue_id)
        atom = ParsedAtom(
            atom_name=atom_name,
            residue_name=residue_name,
            residue_id=residue_id,
            element=element,
            coord=coord,
        )
        atoms.append(atom)
        if residue_name in AROMATIC_RESIDUES and atom_name in AROMATIC_SIDECHAIN_ATOMS.get(residue_name, set()):
            aromatic_groups[residue_label].append(coord)

    aromatic_centroids = {
        residue_label: np.vstack(coords).mean(axis=0)
        for residue_label, coords in aromatic_groups.items()
        if coords
    }
    return atoms, aromatic_centroids


def _parse_pose_atoms(pose_path: Path) -> list[ParsedAtom]:
    atoms: list[ParsedAtom] = []
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip().upper()
        residue_name = line[17:20].strip().upper() or "UNL"
        residue_id = (line[22:26].strip() or "1")
        atom_type = line[77:].strip() if len(line) >= 78 else atom_name
        element = _normalize_element(atom_type or atom_name)
        if element == "H":
            continue
        coord = np.asarray(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ],
            dtype=float,
        )
        atoms.append(
            ParsedAtom(
                atom_name=atom_name,
                residue_name=residue_name,
                residue_id=residue_id,
                element=element,
                coord=coord,
                atom_type=atom_type,
            )
        )
    return atoms


class PoseInteractionAnalyzer:
    def __init__(self, receptor_pdb: Path | None = None):
        self.receptor_atoms, self.aromatic_centroids = _load_receptor_atoms(str(receptor_pdb) if receptor_pdb else None)

    def analyze_pose(self, pose_path: str | Path, smiles: str | None = None) -> dict[str, Any]:
        pose_atoms = _parse_pose_atoms(Path(pose_path))
        if not pose_atoms:
            return {
                "interaction_hbond_count": 0,
                "interaction_hydrophobic_count": 0,
                "interaction_aromatic_count": 0,
                "interaction_salt_bridge_count": 0,
                "interaction_key_residue_count": 0,
                "interaction_key_residues": None,
                "interaction_top_residues": None,
                "interaction_support_score": 0.0,
                "interaction_hinge_contact": False,
                "interaction_summary": None,
            }

        contact_counter: Counter[str] = Counter()
        hbond_residues: set[str] = set()
        hydrophobic_residues: set[str] = set()
        salt_residues: set[str] = set()
        key_residues: set[str] = set()

        ligand_centroid = np.vstack([atom.coord for atom in pose_atoms]).mean(axis=0)
        ligand_aromatic_coords = [atom.coord for atom in pose_atoms if (atom.atom_type or "").strip().upper() == "A"]
        ligand_aromatic_centroid = np.vstack(ligand_aromatic_coords).mean(axis=0) if ligand_aromatic_coords else ligand_centroid

        for ligand_atom in pose_atoms:
            for receptor_atom in self.receptor_atoms:
                dist = _distance(ligand_atom.coord, receptor_atom.coord)
                residue_label = _residue_label(receptor_atom.residue_name, receptor_atom.residue_id)

                if (
                    ligand_atom.element in HBOND_RELEVANT_ELEMENTS
                    and receptor_atom.element in {"N", "O", "S"}
                    and dist <= 3.5
                ):
                    hbond_residues.add(residue_label)
                    contact_counter[residue_label] += 1

                if (
                    ligand_atom.element in HYDROPHOBIC_LIGAND_ELEMENTS
                    and receptor_atom.residue_name in HYDROPHOBIC_RESIDUES
                    and receptor_atom.element in {"C", "S"}
                    and dist <= 4.5
                ):
                    hydrophobic_residues.add(residue_label)
                    contact_counter[residue_label] += 1

                if (
                    ligand_atom.element == "N"
                    and receptor_atom.residue_name in ACIDIC_RESIDUES
                    and receptor_atom.atom_name in ACIDIC_ATOM_NAMES
                    and dist <= 4.0
                ) or (
                    ligand_atom.element == "O"
                    and receptor_atom.residue_name in BASIC_RESIDUES
                    and receptor_atom.atom_name in BASIC_ATOM_NAMES
                    and dist <= 4.0
                ):
                    salt_residues.add(residue_label)
                    contact_counter[residue_label] += 1

                if residue_label in KEY_RESIDUES and dist <= 4.5:
                    key_residues.add(residue_label)
                    contact_counter[residue_label] += 1

        aromatic_residues: set[str] = set()
        for residue_label, centroid in self.aromatic_centroids.items():
            if _distance(ligand_aromatic_centroid, centroid) <= 5.5:
                aromatic_residues.add(residue_label)
                contact_counter[residue_label] += 1

        hinge_contact = any(residue in key_residues for residue in {"ALA743", "LYS745", "MET793"})
        hbond_score = _clip01(len(hbond_residues) / 3.0)
        hydrophobic_score = _clip01(len(hydrophobic_residues) / 6.0)
        aromatic_score = _clip01(len(aromatic_residues) / 2.0)
        salt_score = _clip01(len(salt_residues) / 1.0)
        key_score = _clip01(len(key_residues) / 4.0)
        interaction_support = _clip01(
            0.28 * hbond_score
            + 0.16 * hydrophobic_score
            + 0.14 * aromatic_score
            + 0.12 * salt_score
            + 0.30 * key_score
            + 0.10 * float(hinge_contact)
        )

        top_residues = [residue for residue, _ in contact_counter.most_common(6)]
        summary_bits = []
        if hbond_residues:
            summary_bits.append(f"hbond:{len(hbond_residues)}")
        if aromatic_residues:
            summary_bits.append(f"aromatic:{len(aromatic_residues)}")
        if salt_residues:
            summary_bits.append(f"salt:{len(salt_residues)}")
        if key_residues:
            summary_bits.append(f"key:{','.join(sorted(key_residues))}")

        return {
            "interaction_hbond_count": int(len(hbond_residues)),
            "interaction_hydrophobic_count": int(len(hydrophobic_residues)),
            "interaction_aromatic_count": int(len(aromatic_residues)),
            "interaction_salt_bridge_count": int(len(salt_residues)),
            "interaction_key_residue_count": int(len(key_residues)),
            "interaction_key_residues": ";".join(sorted(key_residues)) if key_residues else None,
            "interaction_top_residues": ";".join(top_residues) if top_residues else None,
            "interaction_support_score": float(interaction_support),
            "interaction_hinge_contact": bool(hinge_contact),
            "interaction_summary": ";".join(summary_bits) if summary_bits else None,
        }
