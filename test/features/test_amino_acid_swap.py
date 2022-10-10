from tempfile import mkstemp
import os
import h5py

from deeprank.features.amino_acid_swap import (
    SIZE_DELTA_FEATURE_NAME,
    CHARGE_DELTA_FEATURE_NAME,
    POLARITY_DELTA_FEATURE_NAME,
    HBACCEPTOR_DELTA_FEATURE_NAME,
    HBDONOR_DELTA_FEATURE_NAME,
    __compute_feature__
)
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import glycine, tryptophan


def test_compute_features():

    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant = PdbVariantSelection("101M", "A", 25, glycine, tryptophan)

    tmp_file, tmp_path = mkstemp(suffix=".h5py")
    os.close(tmp_file)

    try:
        with h5py.File(tmp_path, 'w') as f5:

            xyz_group = f5.require_group('xyz')

            __compute_feature__(environment, 10.0, xyz_group, variant)

            assert SIZE_DELTA_FEATURE_NAME in xyz_group
            assert CHARGE_DELTA_FEATURE_NAME in xyz_group
            assert POLARITY_DELTA_FEATURE_NAME in xyz_group
            assert HBACCEPTOR_DELTA_FEATURE_NAME in xyz_group
            assert HBDONOR_DELTA_FEATURE_NAME in xyz_group
    finally:
        os.remove(tmp_path)
