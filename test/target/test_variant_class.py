import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_, eq_

from deeprank.models.variant import PdbVariantSelection, VariantClass
from deeprank.targets.variant_class import __compute_target__
from deeprank.models.environment import Environment
from deeprank.domain.amino_acid import cysteine, alanine


def test_annotate():
    environment = Environment(pdb_root="test/data/pdb")

    variant = PdbVariantSelection("101M", "A", 10, cysteine, alanine, variant_class=VariantClass.BENIGN)

    work_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(work_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            target_group = f5.require_group("targets")
            __compute_target__(variant, target_group)

            ok_("class" in target_group)
            ok_(target_group["class"], 0.0)
    finally:
        rmtree(work_dir_path)
