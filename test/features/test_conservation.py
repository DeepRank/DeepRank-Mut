import tempfile
import os

from nose.tools import with_setup
import h5py

from deeprank.features import variant_conservation
from deeprank.models.variant import PdbVariantSelection
from deeprank.operate.hdf5data import get_variant_group_name
from deeprank.domain.amino_acid import threonine, lysine


def setup():
    variant_conservation.CONSERVATIONS_DIRECTORY_PATH = "test/data/conservation"


def shutdown():
    pass


@with_setup(setup, shutdown)
def test_conservation():

    variant = PdbVariantSelection("test/data/pdb/1TBG.pdb", "A", 25, threonine, lysine, protein_ac="P16520", protein_residue_number=102)

    hdf5_file, hdf5_path = tempfile.mkstemp(suffix=".hdf5")
    os.close(hdf5_file)

    with h5py.File(hdf5_path, 'w') as f5:

        variant_group = f5.create_group(get_variant_group_name(variant))
        feature_group = variant_group.create_group("conservation_xyz")

        variant_conservation.__compute_feature__(None, feature_group, None, variant)

        wildtype_data = feature_group[variant_conservation.WT_FEATURE_NAME][()]
        variant_data = feature_group[variant_conservation.VAR_FEATURE_NAME][()]

        assert wildtype_data[0, 3] == 0.67529296875, "value is {}".format(wildtype_data[0, 3])
        assert variant_data[0, 3] == 0.0
