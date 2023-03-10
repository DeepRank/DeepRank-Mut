from glob import glob
import os

from pdb2sql import pdb2sql
import numpy

from deeprank.config import logger
from deeprank.features.FeatureClass import FeatureClass
from deeprank.config.chemicals import AA_codes, AA_codes_3to1, AA_codes_1to3
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_pdb_path
from deeprank.parse.pssm import parse_pssm
from deeprank.models.pssm import Pssm
from deeprank.models.residue import Residue
from deeprank.domain.amino_acid import amino_acids


IC_FEATURE_NAME = "residue_information_content"
WT_FEATURE_NAME = "wild_type_probability"
VAR_FEATURE_NAME = "variant_probability"
PSSM_FEATURE_NAME = "pssm_"

def get_neighbour_c_alphas(environment, variant, distance_cutoff):
    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    db = pdb2sql(pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(db, variant.chain_id, variant.residue_number, variant.insertion_code, distance_cutoff):

            # For each residue in the contact range, get the C-alpha:
            for atom in (atom1.residue.atoms + atom2.residue.atoms):
                if atom.name == "CA":
                    atoms.add(atom)

        return atoms
    finally:
        db._close()


def get_c_alpha_pos(environment, variant):
    "gets coordinates for the variant amino acid"

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    db = pdb2sql(pdb_path)
    try:
        if variant.insertion_code is not None:
            position = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, iCode=variant, name="CA")[0]
        else:
            position = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, name="CA")[0]

        return position
    finally:
        db._close()


def get_pssm_paths(pssm_root, pdb_ac):
    """ Finds the PSSM files associated with a PDB entry

        Args:
            pssm_root (str):  path to the directory where the PSSMgen output files are located
            pdb_ac (str): pdb accession code of the entry of interest

        Returns (dict of strings): the PSSM file paths per PDB chain identifier
    """

    paths = glob(os.path.join(pssm_root, "%s/pssm/%s.?.pdb.pssm" % (pdb_ac.lower(), pdb_ac.lower())))
    paths += glob(os.path.join(pssm_root, "%s/%s.?.pdb.pssm" % (pdb_ac.upper(), pdb_ac.upper())))

    return {os.path.basename(path).split('.')[1]: path for path in paths}


def _get_pssm(chain_ids, variant, environment):
    pssm_paths = get_pssm_paths(environment.pssm_root, variant.pdb_ac)

    pssm = Pssm()
    for chain_id in chain_ids:
        if chain_id not in pssm_paths:
            if environment.zero_missing_pssm:
                continue
            else:
                raise FileNotFoundError("No PSSM for {} chain {} in {}".format(variant.pdb_ac, chain_id, environment.pssm_root))

        with open(pssm_paths[chain_id], 'rt', encoding='utf_8') as f:
            pssm.merge_with(parse_pssm(f, chain_id))
    return pssm


def __compute_feature__(environment, distance_cutoff, feature_group, variant):
    "this feature module adds amino acid probability and residue information content as deeprank features"

    # Get the C-alpha atoms, each belongs to a neighbouring residue
    neighbour_c_alphas = get_neighbour_c_alphas(environment, variant, distance_cutoff)

    # Give each chain id a number:
    chain_ids = set([atom.chain_id for atom in neighbour_c_alphas])
    chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

    pssm = _get_pssm(chain_ids, variant, environment)

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")
    feature_object.feature_data_xyz[WT_FEATURE_NAME] = {}
    feature_object.feature_data_xyz[VAR_FEATURE_NAME] = {}
    feature_object.feature_data_xyz[IC_FEATURE_NAME] = {}

    # Get variant probability features and place them at the C-alpha xyz position:
    c_alpha_position = get_c_alpha_pos(environment, variant)
    residue_id = Residue(variant.residue_number, variant.wildtype_amino_acid, variant.chain_id)

    if pssm.has_residue(residue_id):
        wild_type_probability = pssm.get_probability(residue_id, variant.wild_type_amino_acid.code)
        variant_probability = pssm.get_probability(residue_id, variant.variant_amino_acid.code)
        xyz_key = tuple(c_alpha_position)

        feature_object.feature_data_xyz[WT_FEATURE_NAME][xyz_key] = [wild_type_probability]
        feature_object.feature_data_xyz[VAR_FEATURE_NAME][xyz_key] = [variant_probability]

    # For each neighbouring C-alpha, get the residue's PSSM features:
    for atom in neighbour_c_alphas:
        if pssm.has_residue(atom.residue):
            xyz_key = tuple(atom.position)

            feature_object.feature_data_xyz[IC_FEATURE_NAME][xyz_key] = [pssm.get_information_content(atom.residue)]

        for amino_acid in amino_acids:
            feature_name = PSSM_FEATURE_NAME + amino_acid.code
            feature_value = pssm.get_probability(atom.residue, amino_acid.code)

            feature_object.feature_data_xyz[feature_name] = {xyz_key: [feature_value]}

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)

    for key in [WT_FEATURE_NAME, VAR_FEATURE_NAME, IC_FEATURE_NAME]:
        data = numpy.array(feature_group.get(key))
        logger.info("preprocessed {} features for {}:\n{}".format(key, variant, data))
