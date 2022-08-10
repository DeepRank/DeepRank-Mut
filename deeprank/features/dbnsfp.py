from glob import glob
import os

from pdb2sql import pdb2sql
import numpy
import h5py

from deeprank.config import logger
from deeprank.features.FeatureClass import FeatureClass
from deeprank.config.chemicals import AA_codes, AA_codes_3to1, AA_codes_1to3
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_pdb_path
from deeprank.parse.pssm import parse_pssm
from deeprank.models.pssm import Pssm
from deeprank.models.residue import Residue
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection


REVELSCORE_FEATURE_NAME = "revel_score"


def get_c_alpha_pos(environment: Environment, variant: PdbVariantSelection):
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


def __compute_feature__(environment: Environment,
                        distance_cutoff: float,
                        feature_group: h5py.Group,
                        variant: PdbVariantSelection):

    "this feature module extracts REVEL score from an indexed dbNSFP file"


    dbnsfp_path = environment.dbnsfp_path
    if dbnsfp_path is None:
        raise ValueError("dbNSFP hdf5 file not set")

    if variant.protein_residue_number is None:
        raise ValueError("a variant is provided without a protein residue number, thus it cannot be looked up in dbNSFP")

    with h5py.File(dbnsfp_path, 'r') as dbnsfp_file:

        uniprot_group = dbnsfp_file['uniprot']
        enst_group = dbnsfp_file['enst']

        # Look up the protein
        protein_group = None
        if variant.enst_accession is not None:

            if variant.enst_accession in enst_group:
                protein_group = enst_group[variant.enst_accession]

        elif variant.protein_accession is not None:

            if variant.protein_accession in uniprot_group:
                protein_group = uniprot_group[variant.protein_accession]
        else:
            raise ValueError("a variant is provided without a ENST or protein accession, thus it cannot be looked up in dbNSFP")

        if protein_group is None:
            raise ValueError(f"{environment.dbnsfp_path}: no indexed protein found for protein accession {variant.protein_accession} or enst accession {variant.enst_accession}")

        # Look up the residue
        residue_group = protein_group[str(variant.protein_residue_number)]

        # Look up the variant
        variant_group = residue_group[variant.variant_amino_acid.letter]

        # Get the revel score for this variant
        revel_score = variant_group.attrs['revel']

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")

    # Get variant probability features and place them at the C-alpha xyz position:
    c_alpha_position = get_c_alpha_pos(environment, variant)
    residue_id = Residue(variant.residue_number, variant.wildtype_amino_acid, variant.chain_id)
    xyz_key = tuple(c_alpha_position)

    feature_object.feature_data_xyz[REVELSCORE_FEATURE_NAME] = {xyz_key: [revel_score]}

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
