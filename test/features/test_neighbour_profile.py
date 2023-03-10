import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_, eq_

from deeprank.features.neighbour_profile import (__compute_feature__,
                                                 IC_FEATURE_NAME, WT_FEATURE_NAME, VAR_FEATURE_NAME, PSSM_FEATURE_NAME)
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import valine, alanine, glycine, tryptophan, amino_acids
from deeprank.models.environment import Environment


def test_feature():
    environment = Environment(pdb_root="test/data/pdb",
                              pssm_root="test/data/pssm")

    tmp_dir_path = mkdtemp()

    variant = PdbVariantSelection("101M", "A", 25, glycine, tryptophan)

    try:
        hdf5_path = os.path.join(tmp_dir_path, 'test.hdf5')

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")
            __compute_feature__(environment, 10.0, group, variant)

            # Check that the features are present in the list:
            ok_(len(group.get(WT_FEATURE_NAME)) >= 4)
            ok_(len(group.get(VAR_FEATURE_NAME)) >= 4)

            # These features should be collected for ALL neighbouring atoms
            ok_(len(group.get(IC_FEATURE_NAME)) > 20)
            for amino_acid in amino_acids:
                ok_(len(group.get(PSSM_FEATURE_NAME + amino_acid.code)) > 20)
    finally:
        rmtree(tmp_dir_path)


def test_zero_missing_pssm():
    environment = Environment(pdb_root="test/data/pdb",
                              pssm_root="test/data/pssm",
                              zero_missing_pssm=True)

    tmp_dir_path = mkdtemp()

    variant = PdbVariantSelection("1EAU", "A", 16, valine, alanine)

    try:
        hdf5_path = os.path.join(tmp_dir_path, 'test.hdf5')

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")
            __compute_feature__(environment, 10.0, group, variant)

            ok_(len(group.get(WT_FEATURE_NAME)) == 0)
            ok_(len(group.get(IC_FEATURE_NAME)) == 0)
    finally:
        rmtree(tmp_dir_path)
