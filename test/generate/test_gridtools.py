import os
import pkg_resources
from tempfile import mkdtemp, mkstemp
import shutil
import logging

import numpy
import h5py
from nose.tools import ok_, eq_
import pdb2sql

from deeprank.operate import hdf5data
from deeprank.generate.GridTools import GridTools
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.models.variant import PdbVariantSelection
from deeprank.features.atomic_contacts import (__compute_feature__ as compute_contact_feature,
                                               COULOMB_FEATURE_NAME, VANDERWAALS_FEATURE_NAME, CHARGE_FEATURE_NAME)
from deeprank.domain.amino_acid import phenylalanine, tyrosine, valine, aspartate, glutamate, asparagine, unknown_amino_acid


_log = logging.getLogger(__name__)


def _get_feature_grid(hdf5, feature_group_name, feature_name, points_count):

    # Check that the feature grid exists:
    assert feature_group_name in hdf5, \
        "{} not in hdf5, candidates are: {}".format(",".join(hdf5.keys()))
    assert feature_name in hdf5[feature_group_name], \
        "{} not in feature group, candidates are: {}".format(feature_name, ",".join(hdf5[feature_group_name].keys()))

    # Check the grid size
    eq_(hdf5[feature_group_name][feature_name].attrs['sparse'], False)
    eq_(hdf5[feature_group_name][feature_name]['value'].shape, (points_count, points_count, points_count))

    return hdf5[feature_group_name][feature_name]['value']


def gt_(value1, value2):
    assert value1 > value2, "{} <= {}".format(value1, value2)


def lt_(value1, value2):
    assert value1 < value2, "{} >= {}".format(value1, value2)


def test_atomic_contacts_mapping():
    """ In this test, we load an entire pdb entry and generate atomic contacts for it.
        We also mapping these features to the grid and check that this happens correctly.
    """

    pdb_path = "test/101M.pdb"
    variant = PdbVariantSelection(pdb_path, 'A', 138, phenylalanine, tyrosine)
    variant_name = "101M-F138Y"

    feature_types = ["vdwaals", "coulomb", "charge"]

    tmp_dir = mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, "test.hdf5")
        with h5py.File(tmp_path, 'w') as f5:

            variant_group = f5.require_group(variant_name)
            variant_group.attrs['type'] = 'variant'
            hdf5data.store_variant(variant_group, variant)

            feature_group = variant_group.require_group('features')
            raw_feature_group = variant_group.require_group('features_raw')

            compute_contact_feature(pdb_path, feature_group, raw_feature_group, variant)

            for feature_type in feature_types:
                ok_(feature_type in feature_group)

            sqldb = pdb2sql.interface(variant.pdb_path)
            try:
                grid_center = sqldb.get("x,y,z", chainID=variant.chain_id,
                                        resSeq=variant.residue_number, name="CA")[0]
            finally:
                sqldb._close()

            variant_group.require_group('grid_points')
            variant_group['grid_points'].create_dataset('center', data=grid_center)

            points_count = 30

            # Build the grid and map the features.
            gridtools = GridTools(variant_group, variant,
                                  number_of_points=points_count, resolution=1.0,
                                  atomic_densities={'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
                                  feature=feature_types,
                                  contact_distance=8.5,
                                  try_sparse=False)

            for feature_type in feature_types:
                feature_grid = _get_feature_grid(f5,
                                                 "%s/mapped_features/Feature_ind" % variant_name,
                                                 feature_type,
                                                 points_count)

                # Check that the feature is nonzero, at least somewhere on the grid
                ok_(len(numpy.nonzero(feature_grid)) > 0)
    finally:
        shutil.rmtree(tmp_dir)


def test_nan():
    "this variant was reported as NaN-causing"

    pdb_path = "test/data/pdb/5MNH/5MNH.pdb"

    variant = PdbVariantSelection(pdb_path, 'A', 153, aspartate, asparagine)
    variant_name = "NaN-causing"

    feature_types = ["coulomb"]

    tmp_dir = mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, "test.hdf5")
        with h5py.File(tmp_path, 'w') as f5:

            variant_group = f5.require_group(variant_name)
            variant_group.attrs['type'] = 'variant'
            hdf5data.store_variant(variant_group, variant)

            feature_group = variant_group.require_group('features')
            raw_feature_group = variant_group.require_group('features_raw')

            compute_contact_feature(pdb_path, feature_group, raw_feature_group, variant)

            # Build the grid and map the features.
            gridtools = GridTools(variant_group, variant,
                                  number_of_points=20, resolution=1.0,
                                  atomic_densities={'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
                                  feature=feature_types,
                                  contact_distance=8.5,
                                  try_sparse=False)
    finally:
        shutil.rmtree(tmp_dir)


def test_map_negative_contacts_features():
    pdb_path = "test/data/7req.pdb"
    variant = PdbVariantSelection(pdb_path, "A", 255, glutamate, aspartate)

    sqldb = pdb2sql.pdb2sql(pdb_path)
    try:
        mean_position = numpy.mean(sqldb.get("x,y,z"), 0)
    finally:
        sqldb._close()

    hdf5_file, hdf5_path = mkstemp()
    os.close(hdf5_file)

    try:
        with h5py.File(hdf5_path, 'w') as f5:
            variant_group = f5.require_group(str(variant))
            hdf5data.store_variant(variant_group, variant)

            group_xyz = variant_group.require_group("features")
            group_raw = variant_group.require_group("features_raw")

            compute_contact_feature(pdb_path, group_xyz, group_raw, variant)

            augmented_variant_group = f5.require_group(str(variant) + "_r001")
            f5.copy(variant_group.name + '/features/', augmented_variant_group)
            hdf5data.store_variant(augmented_variant_group, variant)

            axis, angle = pdb2sql.transform.get_rot_axis_angle(seed=None)
            DataGenerator._rotate_feature(augmented_variant_group, axis, angle, mean_position)

            for entry_group in [variant_group, augmented_variant_group]:
                GridTools(entry_group, variant,
                         number_of_points=20, resolution=1.0,
                         atomic_densities={'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
                         feature=[COULOMB_FEATURE_NAME, VANDERWAALS_FEATURE_NAME, CHARGE_FEATURE_NAME],
                         contact_distance=8.5,
                         try_sparse=False)

            charges = variant_group["mapped_features/Feature_ind/charge/value"][()]
            vanderwaals = variant_group["mapped_features/Feature_ind/vdwaals/value"][()]
            coulomb = variant_group["mapped_features/Feature_ind/coulomb/value"][()]

            augmented_charges = augmented_variant_group["mapped_features/Feature_ind/charge/value"][()]
            augmented_vanderwaals = augmented_variant_group["mapped_features/Feature_ind/vdwaals/value"][()]
            augmented_coulomb = augmented_variant_group["mapped_features/Feature_ind/coulomb/value"][()]
    finally:
        os.remove(hdf5_path)

    assert len(numpy.nonzero(charges < 0.0)) > 0, "no negative charges in variant group"
    assert len(numpy.nonzero(vanderwaals < 0.0)) > 0, "no negative vanderwaals in variant group"
    assert len(numpy.nonzero(coulomb < 0.0)) > 0, "no negative coulomb in variant group"

    assert len(numpy.nonzero(augmented_charges < 0.0)) > 0, "no negative charges in augmented variant group"
    assert len(numpy.nonzero(augmented_vanderwaals < 0.0)) > 0, "no negative vanderwaals augmented variant group"
    assert len(numpy.nonzero(augmented_coulomb < 0.0)) > 0, "no negative coulomb augmented variant group"

    assert len(numpy.nonzero(charges > 0.0)) > 0, "no positive charges in variant group"
    assert len(numpy.nonzero(vanderwaals > 0.0)) > 0, "no positive vanderwaals in variant group"
    assert len(numpy.nonzero(coulomb > 0.0)) > 0, "no positive coulomb in variant group"

    assert len(numpy.nonzero(augmented_charges > 0.0)) > 0, "no positive charges in augmented variant group"
    assert len(numpy.nonzero(augmented_vanderwaals > 0.0)) > 0, "no positive vanderwaals augmented variant group"
    assert len(numpy.nonzero(augmented_coulomb > 0.0)) > 0, "no positive coulomb augmented variant group"


def test_feature_mapping():
    """ In this test, we investigate a set of five atoms. We make the grid tool take
        the atoms in a 20 A radius around the first atom and compute the carbon density grid around it.
        We're also positioning a feature on a grid and investigate its resulting distribution.

        Grid values should be high close to the set position and low everywhere else.
    """

    pdb_name = "1XXX"


    # Build a temporary directory to store the test file.
    tmp_dir = mkdtemp()

    try:
        pdb_path = os.path.join(tmp_dir, "%s" % pdb_name)

        with open(pdb_path, 'wt') as f:
            for line in [
                "ATOM      1  C   XXX A   1       0.000   0.000   0.000  1.00  0.00      C   C\n",
                "ATOM      2 CA   XXX A   2       1.000   1.000   1.000  1.00  0.00      C   C\n",
                "ATOM      3  N   XXX A   2      -1.000  -1.000  -1.000  1.00  0.00      C   N\n",
                "ATOM      4  N   XXX A   3      10.000  10.000  10.000  1.00  0.00      C   N\n",
                "ATOM      5  C   XXX A   4     -10.000 -10.000 -10.000  1.00  0.00      C   C\n",
            ]:
               f.write(line) 

        variant = PdbVariantSelection(pdb_path, 'A', 1, unknown_amino_acid, valine)
        variant_name = "1XXX-X1V"

        tmp_path = os.path.join(tmp_dir, "test.hdf5")

        with h5py.File(tmp_path, 'w') as f5:

            # Fill the HDF5 with data, before we give it to the grid.
            variant_group = f5.require_group(variant_name)
            variant_group.attrs['type'] = 'variant'
            hdf5data.store_variant(variant_group, variant)

            feature_group = variant_group.require_group('features')

            feature_type_name = "testfeature"
            chain_id = "A"
            chain_number = 0
            position = [10.0, 10.0, 10.0]  # this should fit inside the grid
            value = 0.923

            data = numpy.array([position + [value]])
            feature_group.create_dataset(feature_type_name, data=data)

            points_count = 30

            # Build the grid and map the features.
            gridtools = GridTools(variant_group, variant,
                                  number_of_points=points_count, resolution=1.0,
                                  atomic_densities={'C': 1.7},  # only collect density data on carbon
                                  feature=[feature_type_name],
                                  contact_distance=20.0,  # just take all the atoms, close and far
                                  try_sparse=False)

            carbon_density_grid = _get_feature_grid(f5,
                                                    "%s/mapped_features/AtomicDensities_ind" % variant_name,
                                                    "C",
                                                    points_count)

            # Check that the gaussian is high at the carbon atom positions:
            gt_(carbon_density_grid[14][14][14], 0.1)
            gt_(carbon_density_grid[4][4][4], 0.1)

            # Check that nitrogens are not participating:
            lt_(carbon_density_grid[24][24][24], 0.001)

            feature_grid = _get_feature_grid(f5,
                                             "%s/mapped_features/Feature_ind" % variant_name,
                                             feature_type_name,
                                             points_count)

            # Check that the feature is high at the set position:
            gt_(feature_grid[24][24][24], 0.1)

            # Check that the feature is low where it was not set:
            lt_(feature_grid[0][0][0], 0.001)

    finally:
        # Whether the test completes successfully or not, it needs to clean up after itself.
        shutil.rmtree(tmp_dir)
