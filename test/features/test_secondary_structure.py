import tempfile
import os

import h5py
import numpy

from deeprank.features.secondary_structure import __compute_feature__, SECONDARY_STRUCTURE_FEATURE_NAME
from deeprank.models.variant import PdbVariantSelection
from deeprank.models.environment import Environment
from deeprank.domain.amino_acid import arginine, alanine


def test_secondary_structure_feature_generation():

    test_hdf5_file, test_hdf5_path = tempfile.mkstemp()
    os.close(test_hdf5_file)

    try:
        environment = Environment(dssp_root="test/data/dssp", pdb_root="test/data/pdb")
        variant = PdbVariantSelection("1crn", "A", 10, arginine, alanine)

        with h5py.File(test_hdf5_path, 'w') as test_hdf5_file:

            feature_group = test_hdf5_file.require_group("features_xyz")

            __compute_feature__(environment, 10.0, feature_group, variant)

            feature_data = feature_group[SECONDARY_STRUCTURE_FEATURE_NAME][:]

            # dimension: x,y,z,helix,strand (:, 5)
            assert feature_data.shape[0] > 0, "no feature values"
            assert feature_data.shape[1] == 5, f"feature data shape is {feature_data.shape}"

            # expect some helix/sheet in 1crn
            assert numpy.any(feature_data[:,3:] > 0)
    finally:
        os.remove(test_hdf5_path)
