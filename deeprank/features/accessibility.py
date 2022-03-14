import logging

from pdb2sql import pdb2sql
import freesasa
import numpy

from deeprank.config import logger
from deeprank.features import FeatureClass
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_pdb_path


def get_atoms_of_iterest(environment, variant, distance_cutoff):

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    pdb = pdb2sql(pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(pdb,
                                                           variant.chain_id,
                                                           variant.residue_number,
                                                           variant.insertion_code,
                                                           distance_cutoff):

            # Add all atoms, even from the variant residue itself:
            atoms.add(atom1)
            atoms.add(atom2)

        return atoms
    finally:
        pdb._close()


FEATURE_NAME = "accessibility"


def __compute_feature__(environment, distance_cutoff, featgrp, variant):
    "computes SASA-based features"

    # Let pdb2sql tell us which atoms are around the variant residue:
    atoms_keys = set([])
    chain_ids = set([])
    for atom in get_atoms_of_iterest(environment, variant, distance_cutoff):
        residue_id = str(atom.residue.number)
        if variant.insertion_code is not None:
            residue_id += variant.insertion_code
        atom_key = (atom.chain_id.strip(), residue_id, atom.name.strip())
        atoms_keys.add(atom_key)
        chain_ids.add(atom.chain_id)

        logger.debug("contact atom: {}".format(atom_key))

    # Get structure and area data from SASA:
    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)

    # Give each chain id a number:
    chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

    # Prepare a deeprank feature object:
    feature_obj = FeatureClass('Atomic')

    feature_obj.feature_data_xyz[FEATURE_NAME] = {}

    # Iterate over atoms in SASA:
    for atom_index in range(structure.nAtoms()):

        # Get atom info from SASA:
        position = structure.coord(atom_index)
        chain_id = structure.chainLabel(atom_index)
        atom_key = (chain_id.strip(),
                    structure.residueNumber(atom_index).strip(),
                    structure.atomName(atom_index).strip())

        logger.debug("atom {}: {}".format(atom_index, atom_key))

        # Check that the atom is one of the selected atoms:
        if atom_key in atoms_keys:

            # Store the accessibility as a feature:
            area = result.atomArea(atom_index)

            logger.debug("  is contact atom with area = {} square Angstrom".format(area))

            xyz_key = tuple(position)
            feature_obj.feature_data_xyz[FEATURE_NAME][xyz_key] = [area]

    # Store the features in the hdf5 file:
    feature_obj.export_dataxyz_hdf5(featgrp)

    data = numpy.array(featgrp.get(FEATURE_NAME))
    logger.info("preprocessed {} features for {}:\n{}".format(FEATURE_NAME, variant, data))
