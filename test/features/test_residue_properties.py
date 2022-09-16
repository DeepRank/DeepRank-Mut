from tempfile import mkstemp
import os

import h5py
import numpy

from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import glycine, tryptophan
from deeprank.models.environment import Environment
from deeprank.features.residue_properties import __compute_feature__, BFACTOR_FEATURE_NAME


def test_feature():
    environment = Environment(pdb_root="test/data/pdb")

    variant = PdbVariantSelection("101M", "A", 25, glycine, tryptophan)

    hdf5_file, hdf5_path = mkstemp(suffix="hdf5")
    os.close(hdf5_file)

    try:
        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")

            __compute_feature__(environment, 10.0, group, variant)

            # Check that the features are present on the grid:
            assert numpy.size(group.get(BFACTOR_FEATURE_NAME)) > 0
            assert numpy.any(group.get(BFACTOR_FEATURE_NAME)[3:] > 0.0)
    finally:
        os.remove(hdf5_path)
