import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_, eq_

from deeprank.features.neighbour_profile import (__compute_feature__,
                                                 IC_FEATURE_NAME, WT_FEATURE_NAME, VAR_FEATURE_NAME)
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import valine, alanine, glycine, tryptophan
from deeprank.models.environment import Environment


def test_feature():
    environment = Environment(pdb_root="test/data/pdb",
                              pssm_root="test/data/pssm")

    tmp_dir_path = mkdtemp()

    try:
        hdf5_path = os.path.join(tmp_dir_path, 'test.hdf5')

        variant = PdbVariantSelection("101M", "A", 25, glycine, tryptophan)

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")
            __compute_feature__(environment, group, None, variant)

            # Check that the features are present on the grid:
            ok_(len(group.get(WT_FEATURE_NAME)) > 0)
            ok_(len(group.get(VAR_FEATURE_NAME)) > 0)
            ok_(len(group.get(IC_FEATURE_NAME)) > 0)
    finally:
        rmtree(tmp_dir_path)


