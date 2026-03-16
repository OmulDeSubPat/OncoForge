from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw


ATOM_COLORS = {
    1: "#d9d9d9",
    6: "#4d4d4d",
    7: "#3b82f6",
    8: "#ef4444",
    9: "#22c55e",
    15: "#f59e0b",
    16: "#facc15",
    17: "#10b981",
    35: "#b45309",
    53: "#7c3aed",
}


def _candidate_legend(row: pd.Series) -> str:
    return (
        f"Rank {int(row.get('rank', 0)) if pd.notna(row.get('rank', None)) else '-'}\n"
        f"Score={float(row.get('final_score', 0.0)):.2f}\n"
        f"pIC50={float(row.get('predicted_pIC50', 0.0)):.2f}\n"
        f"QED={float(row.get('QED', 0.0)):.2f}"
    )


def build_candidate_grid(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> Path | None:
    subset = df.head(top_n).copy()
    mols = []
    legends = []
    for _, row in subset.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(_candidate_legend(row))

    if not mols:
        return None

    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(360, 300),
        legends=legends,
        useSVG=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path))
    return out_path


def _embed_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    with rdBase.BlockLogs():
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            return None
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    return mol


def _embed_from_pose(smiles: str, pose_path: str | None):
    if not pose_path:
        return None
    pose_file = Path(pose_path)
    if not pose_file.exists():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    coords = []
    for line in pose_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_type = line[77:].strip() if len(line) >= 78 else line[12:16].strip()
        if atom_type.upper().startswith("H"):
            continue
        coords.append(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
        )

    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(coords) != len(heavy_indices):
        return None

    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))
    for atom_idx, coord in zip(heavy_indices, coords):
        conf.SetAtomPosition(atom_idx, tuple(float(value) for value in coord))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def _render_single_3d(ax, mol, elev: float, azim: float) -> None:
    conf = mol.GetConformer()
    coords = conf.GetPositions()

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        xyz = coords[[begin, end]]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#7c7c7c", linewidth=1.6, alpha=0.85)

    atom_colors = [ATOM_COLORS.get(atom.GetAtomicNum(), "#6b7280") for atom in mol.GetAtoms()]
    atom_sizes = [60 if atom.GetAtomicNum() > 1 else 24 for atom in mol.GetAtoms()]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=atom_colors, s=atom_sizes, depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def build_candidate_3d_views(row: pd.Series, out_path: Path) -> Path | None:
    mol = _embed_from_pose(str(row["smiles"]), str(row.get("docking_pose_path", ""))) or _embed_3d(str(row["smiles"]))
    if mol is None:
        return None

    fig = plt.figure(figsize=(11, 3.8))
    views = [(20, 35), (20, 125), (75, 45)]
    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        _render_single_3d(ax, mol, elev=elev, azim=azim)

    score = float(row.get("final_score", 0.0))
    pic50 = float(row.get("predicted_pIC50", 0.0))
    qed = float(row.get("QED", 0.0))
    rank = row.get("rank", "")
    fig.suptitle(f"Candidate {rank} | Score {score:.2f} | pIC50 {pic50:.2f} | QED {qed:.2f}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
