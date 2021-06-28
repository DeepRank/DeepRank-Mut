from tempfile import mkdtemp
from shutil import rmtree
import os
import logging

from nose.tools import ok_
import h5py
from pdb2sql import pdb2sql
import numpy

from deeprank.models.variant import PdbVariantSelection
from deeprank.features.accessibility import __compute_feature__, FEATURE_NAME


_log = logging.getLogger(__name__)


def _find_feature_value_by_xyz(position, data):
    key = list(position)
    for row in data:
        if list(row[:3]) == key:
            return row[3]

    raise ValueError("Not found: {}".format(position))


def test_compute_feature():

    pdb_path = "test/101M.pdb"

    tmp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            feature_group = f5.require_group("features")
            raw_feature_group = f5.require_group("raw_features")
            variant = PdbVariantSelection(pdb_path, 'A', 17, 'T')

            __compute_feature__(variant.pdb_path, feature_group, raw_feature_group, variant)

            data = feature_group.get(FEATURE_NAME)

            # Did the feature get stored:
            ok_(len(data) > 0)

            # There must be buried atoms:
            ok_(any([row[3] == 0.0 for row in data]))

            # There must be accessible atoms:
            ok_(any([row[3] > 0.0 for row in data]))

            pdb = pdb2sql(pdb_path)
            try:
                # Atoms at the surface should have higher SASA than buried atoms
                position = pdb.get("x,y,z", resSeq=19, resName="ALA", name="CB")[0]
                surface_sasa = _find_feature_value_by_xyz(position, data)

                position = pdb.get("x,y,z", resSeq=111, resName="ILE", name="CG1")[0]
                buried_sasa = _find_feature_value_by_xyz(position, data)

                ok_(surface_sasa > buried_sasa)
            finally:
                pdb._close()
    finally:
        rmtree(tmp_dir_path)
