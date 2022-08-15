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


AF_FEATURE_NAME = "allele_frequency"


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

    "this feature module extracts allele frequency from an indexed gnomAD file"


    gnomad_path = environment.gnomad_path
    if gnomad_path is None:
        raise ValueError("gnomAD hdf5 file not set")

    if variant.protein_residue_number is None:
        raise ValueError("a variant is provided without a protein residue number, thus it cannot be looked up in gnomAD")

    if variant.enst_accession is None:
        raise ValueError("a variant is provided without a protein residue number, thus it cannot be looked up in gnomAD")

    variant_name = f"{variant.enst_accession}{variant.wildtype_amino_acid.code.upper()}{variant.protein_residue_number}{variant.variant_amino_acid.code.upper()}"

    with h5py.File(gnomad_path, 'r') as gnomad_file:

        if variant_name not in gnomad_file:
            raise ValueError(f"{variant_name} wasn't found in {gnomad_path}")

        # Look up the variant
        variant_group = gnomad_file[variant_name]

        # Get the allele frequency for this variant
        allele_frequency = variant_group.attrs['AF']

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")

    # Get variant probability features and place them at the C-alpha xyz position:
    c_alpha_position = get_c_alpha_pos(environment, variant)
    residue_id = Residue(variant.residue_number, variant.wildtype_amino_acid, variant.chain_id)
    xyz_key = tuple(c_alpha_position)

    feature_object.feature_data_xyz[AF_FEATURE_NAME] = {xyz_key: [allele_frequency]}

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
