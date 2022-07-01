import tempfile
import os

from nose.tools import with_setup
import h5py
import numpy

from deeprank.features import variant_conservation
from deeprank.models.variant import PdbVariantSelection
from deeprank.operate.hdf5data import get_variant_group_name
from deeprank.domain.amino_acid import threonine, lysine, alanine, aspartate
from deeprank.models.environment import Environment


def test_conservation():

    environment = Environment(pdb_root="test/data/pdb/",
                              conservation_root="test/data/conservation")

    variant = PdbVariantSelection("1TBG", "A", 25, threonine, lysine, protein_accession="P16520", protein_residue_number=102)

    hdf5_file, hdf5_path = tempfile.mkstemp(suffix=".hdf5")
    os.close(hdf5_file)

    try:
        with h5py.File(hdf5_path, 'w') as f5:

            variant_group = f5.create_group(get_variant_group_name(variant))
            feature_group = variant_group.create_group("conservation_xyz")

            variant_conservation.__compute_feature__(environment, 10.0, feature_group, variant)

            wildtype_data = feature_group[variant_conservation.WT_FEATURE_NAME][()]
            variant_data = feature_group[variant_conservation.VAR_FEATURE_NAME][()]

            # There should be a whole list of positions and feature values, not just for the C-alpha
            assert wildtype_data.shape[0] > 1
            assert wildtype_data.shape[1] == 4

            assert wildtype_data[0, 3] == 0.67529296875, "value is {}".format(wildtype_data[0, 3])
            assert variant_data[0, 3] == 0.0
    finally:
        os.remove(hdf5_path)


def test_conservation_nan():
    "test on a dataset with NaN values"

    environment = Environment(pdb_root="test/data/pdb/",
                              conservation_root="test/data/conservation")

    variant = PdbVariantSelection("1QUU", "A", 76, alanine, aspartate, protein_accession="P35609", protein_residue_number=1)

    hdf5_file, hdf5_path = tempfile.mkstemp(suffix=".hdf5")
    os.close(hdf5_file)

    try:
        with h5py.File(hdf5_path, 'w') as f5:

            variant_group = f5.create_group(get_variant_group_name(variant))
            feature_group = variant_group.create_group("conservation_xyz")

            variant_conservation.__compute_feature__(environment, 10.0, feature_group, variant)

            wildtype_data = feature_group[variant_conservation.WT_FEATURE_NAME][()]
            variant_data = feature_group[variant_conservation.VAR_FEATURE_NAME][()]

            assert not numpy.any(numpy.isnan(wildtype_data)), "wildtype is NaN"
            assert not numpy.any(numpy.isnan(variant_data)), "variant is NaN"
    finally:
        os.remove(hdf5_path)


