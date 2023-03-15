from glob import glob
import os
from typing import Dict, Set

from pdb2sql import pdb2sql
import numpy
import h5py

from deeprank.config import logger
from deeprank.features.FeatureClass import FeatureClass
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_pdb_path
from deeprank.parse.dssp import parse_dssp
from deeprank.models.residue import Residue
from deeprank.models.atom import Atom
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.models.secondary_structure import SecondaryStructure


SECONDARY_STRUCTURE_FEATURE_NAME = "secondary_structure"


def _get_neighbour_atoms(environment: Environment, variant: PdbVariantSelection, distance_cutoff: float) -> Set[Atom]:
    "gets all atoms in a radius around the variant residue"

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    db = pdb2sql(pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(db, variant.chain_id, variant.residue_number, variant.insertion_code, distance_cutoff):

            # For each residue in the contact range, get the C-alpha:
            for atom in (atom1.residue.atoms + atom2.residue.atoms):
                atoms.add(atom1)

        return atoms
    finally:
        db._close()


def _get_dssp(variant: PdbVariantSelection, environment: Environment) -> Dict[Residue, SecondaryStructure]:
    "gets the dssp data for the entire structure"

    dssp_path = os.path.join(environment.dssp_root, f"{variant.pdb_ac.lower()}.dssp")

    dssp = parse_dssp(dssp_path)

    return dssp


def __compute_feature__(environment: Environment, distance_cutoff: float, feature_group: h5py.Group, variant: PdbVariantSelection):
    "this feature module adds secondary structure as deeprank features"

    # Get the atoms, each belongs to a neighbouring residue
    atoms = _get_neighbour_atoms(environment, variant, distance_cutoff)

    # get the secondary structure data
    dssp = _get_dssp(variant, environment)

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")

    # Fore each atom, take its residue's secondary structure data.
    feature_object.feature_data_xyz[SECONDARY_STRUCTURE_FEATURE_NAME] = {}
    for atom in atoms:
        xyz_key = tuple(atom.position)

        if atom.residue not in dssp:
            raise ValueError(f"{atom.residue} is missing from the dssp data")

        secondary_structure = dssp[atom.residue]

        feature_object.feature_data_xyz[SECONDARY_STRUCTURE_FEATURE_NAME][xyz_key] = secondary_structure.one_hot()

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)

