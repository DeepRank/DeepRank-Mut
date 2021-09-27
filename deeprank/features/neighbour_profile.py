from pdb2sql import pdb2sql
import numpy

from deeprank.config import logger
from deeprank.features.FeatureClass import FeatureClass
from deeprank.config.chemicals import AA_codes, AA_codes_3to1, AA_codes_1to3
from deeprank.operate.pdb import get_residue_contact_atom_pairs
from deeprank.parse.pssm import parse_pssm
from deeprank.models.pssm import Pssm
from deeprank.models.residue import Residue


IC_FEATURE_NAME = "residue_information_content"
WT_FEATURE_NAME = "wild_type_probability"
VAR_FEATURE_NAME = "variant_probability"

def get_neighbour_c_alphas(variant, distance_cutoff):
    db = pdb2sql(variant.pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(db, variant.chain_id, variant.residue_number, distance_cutoff):

            # For each residue in the contact range, get the C-alpha:
            for atom in (atom1.residue.atoms + atom2.residue.atoms):
                if atom.name == "CA":
                    atoms.add(atom1)

        return atoms
    finally:
        db._close()


def get_c_alpha_pos(variant):
    db = pdb2sql(variant.pdb_path)
    try:
        position = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, name="CA")[0]

        return position
    finally:
        db._close()


def get_wild_type_amino_acid(variant):
    db = pdb2sql(variant.pdb_path)
    try:
        residue_names = db.get("resName", chainID=variant.chain_id, resSeq=variant.residue_number)
        if len(residue_names) == 0:
            raise ValueError("no residue {} {} in {}"
                             .format(variant.chain_id, variant.residue_number, variant.pdb_path))

        amino_acid_code = residue_names[0]

        return amino_acid_code
    finally:
        db._close()


def _get_pssm(chain_ids, variant):
    pssm = Pssm()
    for chain_id in chain_ids:
        with open(variant.get_pssm_path(chain_id), 'rt', encoding='utf_8') as f:
            pssm.merge_with(parse_pssm(f, chain_id))
    return pssm


def __compute_feature__(pdb_data, feature_group, raw_feature_group, variant):
    "this feature module adds amino acid probability and residue information content as deeprank features"

    # Get the C-alpha atoms, each belongs to a neighbouring residue
    neighbour_c_alphas = get_neighbour_c_alphas(variant, 10.0)

    # Give each chain id a number:
    chain_ids = set([atom.chain_id for atom in neighbour_c_alphas])
    chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

    pssm = _get_pssm(chain_ids,variant)

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")

    # Get variant probability features and place them at the C-alpha xyz position:
    c_alpha_position = get_c_alpha_pos(variant)
    wild_type_code = get_wild_type_amino_acid(variant)
    residue_id = Residue(variant.residue_number, wild_type_code, variant.chain_id)
    wild_type_probability = pssm.get_probability(residue_id, variant.wild_type_amino_acid.code)
    variant_probability = pssm.get_probability(residue_id, variant.variant_amino_acid.code)
    xyz_key = tuple(c_alpha_position)

    feature_object.feature_data_xyz[WT_FEATURE_NAME] = {xyz_key: [wild_type_probability]}
    feature_object.feature_data_xyz[VAR_FEATURE_NAME] = {xyz_key: [variant_probability]}

    # For each neighbouring C-alpha, get the residue's PSSM features:
    feature_object.feature_data_xyz[IC_FEATURE_NAME] = {}
    for atom in neighbour_c_alphas:
        xyz_key = tuple(atom.position)

        feature_object.feature_data_xyz[IC_FEATURE_NAME][xyz_key] = [pssm.get_information_content(atom.residue)]

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)

    for key in [WT_FEATURE_NAME, VAR_FEATURE_NAME, IC_FEATURE_NAME]:
        data = numpy.array(feature_group.get(key))
        logger.info("preprocessed {} features for {}:\n{}".format(key, variant, data))
