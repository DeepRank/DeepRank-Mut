import os
import sys
import unittest
from time import time
import shutil
from tempfile import mkdtemp
import logging

from unittest.mock import patch, MagicMock
import numpy
import h5py
from nose.tools import eq_, ok_

from deeprank.generate import *
from deeprank.models.variant import PdbVariantSelection
from deeprank.tools.sparse import FLANgrid
from deeprank.operate import hdf5data
from deeprank.domain.amino_acid import glycine, alanine, asparagine


_log = logging.getLogger(__name__)


def test_generate():
    """ This unit test checks that the HDF5 preprocessing code generates the correct data.
        It doesn't use any of the actual feature classes, but instead uses some simple test classes
        that have been created for this purpose.

        Data is expected to be mapped onto a 3D grid.
    """

    number_of_points = 30
    resolution = 1.0
    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
        'number_of_points': [number_of_points, number_of_points, number_of_points],
        'resolution': [resolution, resolution, resolution],
        'atomic_densities': atomic_densities,
    }

    # These classes are made for testing, they give meaningless numbers.
    feature_names = ["test.feature.feature1", "test.feature.feature2"]
    target_names = ["test.target.target1"]

    pdb_path = "test/101m.pdb"
    variants = [PdbVariantSelection("test/101m.pdb", 'A', 25, glycine, alanine),
                PdbVariantSelection("test/data/1eau.pdb", 'A', 72, asparagine, alanine)]

    tmp_dir = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir, "data.hdf5")

        # Make the class put the data in the HDF5 file:
        data_generator = DataGenerator(variants, None, target_names, feature_names, 1, hdf5_path)
        data_generator.create_database()
        data_generator.map_features(grid_info)

        # Read the resulting HDF5 file and check for all variant data.
        with h5py.File(hdf5_path, 'r') as f5:
            eq_(list(f5.attrs['targets']), target_names)
            eq_(list(f5.attrs['features']), feature_names)

            for variant in variants:
                variant_name = hdf5data.get_variant_group_name(variant)

                # Check that the right number of grid point coordinates have been stored and are equally away from each other:
                for coord in ['x', 'y', 'z']:
                    coords = f5["%s/grid_points/%s" % (variant_name, coord)]
                    for i in range(number_of_points - 1):
                        eq_(coords[i + 1] - coords[i], resolution)

                # Check for mapped features in the HDF5 file:
                for map_name in ["Feature_ind", "AtomicDensities_ind"]:
                    features = hdf5data.load_grid_data(f5[variant_name], "Feature_ind")
                    ok_(len(features) > 0)
                    for feature_name, feature_data in features.items():
                        ok_(feature_data.shape == (number_of_points, number_of_points, number_of_points))

                # Check that the target values have been placed in the HDF5 file:
                for target_name in target_names:
                    target_name = target_name.split('.')[-1]
                    target_path = "%s/targets" % variant_name
                    ok_(target_name in f5[target_path])
    finally:
        shutil.rmtree(tmp_dir)


def test_skip_error():
    "preprocessing should continue if only one variant raises an error"

    tmp_dir = mkdtemp()

    # Use one correct pdb file and one wrong pdb file
    variants = [PdbVariantSelection("test/101m.pdb", 'A', 25, glycine, alanine),
                PdbVariantSelection("test/data/wrng.pdb", "A", 1, glycine, alanine)]

    # These classes are made for testing, they give meaningless numbers.
    feature_names = ["test.feature.feature1", "test.feature.feature2"]
    target_names = ["test.target.target1"]

    number_of_points = 30
    resolution = 1.0
    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
        'number_of_points': [number_of_points, number_of_points, number_of_points],
        'resolution': [resolution, resolution, resolution],
        'atomic_densities': atomic_densities,
    }

    hdf5_path = os.path.join(tmp_dir, "test.hdf5")

    try:
        data_generator = DataGenerator(variants, None, target_names, feature_names, 1, hdf5_path)
        data_generator.create_database()
        data_generator.map_features(grid_info)

        # Check that the output file is not empty
        with h5py.File(hdf5_path, 'r') as f5:
            ok_(len(f5.keys()) > 0)

    finally:
        shutil.rmtree(tmp_dir)


@patch("deeprank.generate.GridTools.map_features")
def test_skip_nan(mock_map_features):
    "NaN features should not be added to the preprocessing"

    number_of_points = 20

    nan_dict = {}
    for feature_name in ["feature1", "feature2"]:

        grid = numpy.empty((number_of_points, number_of_points, number_of_points))
        grid[:] = numpy.nan

        nan_dict[feature_name] = grid

    mock_map_features.return_value = nan_dict

    tmp_dir = mkdtemp()

    variants = [PdbVariantSelection("test/101m.pdb", 'A', 25, glycine, alanine)]

    feature_names = ["test.feature.feature1", "test.feature.feature2"]
    target_names = ["test.target.target1"]

    resolution = 1.0
    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
        'number_of_points': [number_of_points, number_of_points, number_of_points],
        'resolution': [resolution, resolution, resolution],
        'atomic_densities': atomic_densities,
    }

    hdf5_path = os.path.join(tmp_dir, "test.hdf5")

    try:
        data_generator = DataGenerator(variants, None, target_names, feature_names, 1, hdf5_path)
        data_generator.create_database()
        data_generator.map_features(grid_info)

        # Check that the output file is empty, since nan entries don't belong in the output.
        with h5py.File(hdf5_path, 'r') as f5:
            ok_(len(f5.keys()) == 0)
    finally:
        shutil.rmtree(tmp_dir)
