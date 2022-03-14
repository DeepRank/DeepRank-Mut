from tempfile import mkdtemp
from shutil import rmtree
import os
import logging

from nose.tools import ok_
import h5py
from pdb2sql import pdb2sql
import numpy

from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.features.accessibility import __compute_feature__, FEATURE_NAME
from deeprank.operate.pdb import get_pdb_path
from deeprank.domain.amino_acid import valine, threonine, serine, cysteine


_log = logging.getLogger(__name__)


def _find_feature_value_by_xyz(position, data):
    key = list(position)
    for row in data:
        if list(row[:3]) == key:
            return row[3]

    raise ValueError("Not found: {}".format(position))


def test_compute_feature():

    environment = Environment(pdb_root="test/data/pdb/")

    tmp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            feature_group = f5.require_group("features")
            variant = PdbVariantSelection("101M", 'A', 17, valine, threonine)

            __compute_feature__(environment, 10.0, feature_group, variant)

            data = feature_group.get(FEATURE_NAME)

            # Did the feature get stored:
            ok_(len(data) > 0)

            # There must be buried atoms:
            ok_(any([row[3] == 0.0 for row in data]))

            # There must be accessible atoms:
            ok_(any([row[3] > 0.0 for row in data]))

            pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)
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

def test_compute_feature_with_altlocs():

    environment = Environment(pdb_root="test/data/pdb")

    tmp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            feature_group = f5.require_group("features")
            variant = PdbVariantSelection("5EYU", 'A', 8, serine, cysteine)

            __compute_feature__(environment, 10.0, feature_group, variant)

            data = feature_group.get(FEATURE_NAME)

            # Did the feature get stored:
            ok_(len(data) > 0)

            # Any NaN values:
            ok_(not numpy.any(numpy.isnan(data)))
    finally:
        rmtree(tmp_dir_path)
