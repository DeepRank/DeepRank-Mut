from tempfile import mkstemp
import os

import h5py

from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import *
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.models.environment import Environment


def test_get_grid_center():
    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant = PdbVariantSelection("101m", "A", 25, glycine, alanine)

    DataGenerator.get_grid_center(environment, variant)

def test_rot_axis():
    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant_nmr = PdbVariantSelection("1CR4", "A", 10, aspartate, glutamate)
    variant_pdb = PdbVariantSelection("101M", "A", 25, glycine, alanine)

    dg = DataGenerator(environment, [variant_nmr, variant_pdb])

    for variant in [variant_nmr, variant_pdb]:

        tmp_file, tmp_path = mkstemp()
        os.close(tmp_file)
        try:
            with h5py.File(tmp_path, 'w') as f5:
                dg._add_aug_pdb(f5, variant, "var", [1., 0., 0.], 0.1)
        finally:
            os.remove(tmp_path)
