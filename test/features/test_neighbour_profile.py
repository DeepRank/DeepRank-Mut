import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_, eq_

from deeprank.features.neighbour_profile import (__compute_feature__, get_wild_type_amino_acid,
                                                 IC_FEATURE_NAME, WT_FEATURE_NAME, VAR_FEATURE_NAME)
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import valine, alanine, glycine, tryptophan


def test_feature():
    tmp_dir_path = mkdtemp()

    try:
        hdf5_path = os.path.join(tmp_dir_path, 'test.hdf5')

        variant = PdbVariantSelection("test/101M.pdb", "A", 25, glycine, tryptophan,
                                      {'A': "test/101M.A.pdb.pssm"})

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")
            __compute_feature__(variant.pdb_path, group, None, variant)

            # Check that the features are present on the grid:
            ok_(len(group.get(WT_FEATURE_NAME)) > 0)
            ok_(len(group.get(VAR_FEATURE_NAME)) > 0)
            ok_(len(group.get(IC_FEATURE_NAME)) > 0)
    finally:
        rmtree(tmp_dir_path)


def test_wt():
    variant = PdbVariantSelection("test/5unf.pdb", "B", 164, valine, alanine)
    wt = get_wild_type_amino_acid(variant)
    eq_(wt, "VAL")
