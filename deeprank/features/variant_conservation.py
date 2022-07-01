import os

import numpy
from pdb2sql import pdb2sql
import pandas

from deeprank.features.FeatureClass import FeatureClass
from deeprank.operate.pdb import get_pdb_path


def _get_atom_positions(pdb_root, variant):

    pdb_path = get_pdb_path(pdb_root, variant.pdb_ac)

    db = pdb2sql(pdb_path)
    try:
        if variant.insertion_code is not None:

            positions = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, iCode=variant.insertion_code)
        else:
            positions = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number)

        return positions
    finally:
        db._close()


def _get_conservations_dataframe(conservation_root, protein_accession):

    for path in [os.path.join(conservation_root, "{}.parq".format(protein_accession)),
                 os.path.join(conservation_root, protein_accession, "{}.parq".format(protein_accession))]:

        if os.path.isfile(path):
            return pandas.read_parquet(path)

    raise FileNotFoundError("No conservations file for {}".format(protein_accession))


WT_FEATURE_NAME = "wildtype_conservation"
VAR_FEATURE_NAME = "variant_conservation"


def __compute_feature__(environment, distance_cutoff, feature_group, variant):
    "this feature module adds amino acid conservation as deeprank features"

    if variant.protein_accession is None or variant.protein_residue_number is None:
        raise RuntimeError("protein accession and protein residue number must be set on the variant to generate this feature")

    # Get conservations data from file:
    conservations_dataframe = _get_conservations_dataframe(environment.conservation_root, variant.protein_accession)
    residue_row = conservations_dataframe.iloc[variant.protein_residue_number - 1]
    wildtype_conservation = residue_row["sub_consv_{}".format(variant.wildtype_amino_acid.letter)]
    variant_conservation = residue_row["sub_consv_{}".format(variant.variant_amino_acid.letter)]

    # make sure there are no NaNs:
    if numpy.isnan(wildtype_conservation):
        wildtype_conservation = 0.0

    if numpy.isnan(variant_conservation):
        variant_conservation = 0.0

    feature_object = FeatureClass("Residue")
    feature_object.feature_data_xyz[WT_FEATURE_NAME] = {}
    feature_object.feature_data_xyz[VAR_FEATURE_NAME] = {}

    # Get the C-alpha position to store the feature with:
    positions = _get_atom_positions(environment.pdb_root, variant)
    for position in positions:
        xyz_key = tuple(position)

        # Store features:
        feature_object.feature_data_xyz[WT_FEATURE_NAME][xyz_key] = [wildtype_conservation]
        feature_object.feature_data_xyz[VAR_FEATURE_NAME][xyz_key] = [variant_conservation]

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
