import h5py
from pdb2sql import pdb2sql

from deeprank.features.FeatureClass import FeatureClass
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_pdb_path


BFACTOR_FEATURE_NAME = "b_factor"


def get_neighbour_residues(environment, variant, distance_cutoff):
    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    db = pdb2sql(pdb_path)
    try:
        residues = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(db, variant.chain_id, variant.residue_number, variant.insertion_code, distance_cutoff):

            residues.add(atom1.residue)
            residues.add(atom2.residue)

        return residues
    finally:
        db._close()



def __compute_feature__(environment: Environment,
                        distance_cutoff: float,
                        feature_group: h5py.Group,
                        variant: PdbVariantSelection):
    """ Included with this feature module:
         - b_factor: residue average B-factors
    """

    # Fetch the residues in the neighbourhood:
    residues = get_neighbour_residues(environment, variant, distance_cutoff)

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")

    feature_object.feature_data_xyz[BFACTOR_FEATURE_NAME] = {}

    for residue in residues:
        b_factor_average = sum([atom.b_factor for atom in residue.atoms]) / len(residue.atoms)

        for atom in residue.atoms:
            xyz = tuple(atom.position)
            feature_object.feature_data_xyz[BFACTOR_FEATURE_NAME][xyz] = [b_factor_average]

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
