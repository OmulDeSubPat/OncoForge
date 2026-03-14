from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_NAMES = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "aromatic_rings",
    "fraction_csp3",
    "heavy_atoms",
    "hetero_atoms",
    "mol_mr",
    "valence_electrons",
]


def descriptor_vector_from_mol(mol) -> np.ndarray:
    if mol is None:
        raise ValueError("Invalid molecule")

    values = np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            Crippen.MolMR(mol),
            Descriptors.NumValenceElectrons(mol),
        ],
        dtype=np.float32,
    )
    return values


def descriptor_vector_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return descriptor_vector_from_mol(mol)
