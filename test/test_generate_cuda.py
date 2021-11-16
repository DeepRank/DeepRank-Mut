import os
import sys
import unittest
from time import time

from deeprank.generate import *
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import leucine, alanine, histidine

try:
    import pycuda
    skip = False
except BaseException:
    skip = True


class TestGenerateCUDA(unittest.TestCase):

    tune = False
    test = False
    gpu_block = [8, 8, 8]
    h5file = '1ak4_cuda.hdf5'

    environment = Environment(pdb_root="test/data/pdb",
                              pssm_root="test/data/pssm")

    variants = [PdbVariantSelection("101M", "A", 9, leucine, alanine),
                PdbVariantSelection("101M", "A", 12, histidine, alanine)]


    @unittest.skipIf(skip, "torch fails on Travis")
    def test_generate_cuda(self):

        # init the data assembler
        database = DataGenerator(environment, variants, None,
            compute_targets=['deeprank.targets.dockQ'],
            compute_features=[
                'deeprank.features.atomic_contacts',
                'deeprank.features.neighbour_profile',
                'deeprank.features.accessibility'],
            hdf5=self.h5file)

        # map the features
        grid_info = {
            'number_of_points': [30, 30, 30],
            'resolution': [1., 1., 1.],
            'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
        }

        # tune the kernel
        if self.tune:
            database.tune_cuda_kernel(grid_info, func='gaussian')
        # test thekernel
        elif self.test:
            database.test_cuda(grid_info, self.gpu_block, func='gaussian')
        # compute features
        else:

            # create new files
            if not os.path.isfile(database.hdf5):
                t0 = time()
                print('\nCreate new database: %s' % database.hdf5)
                database.create_database()
                print('--> Done in %f s.' % (time() - t0))
            else:
                print('\nUse existing database: %s' % database.hdf5)

            # map these data
            t0 = time()
            print('\nMap features in database: %s' % database.hdf5)
            database.map_features(
                grid_info,
                try_sparse=True,
                time=False,
                cuda=True,
                gpu_block=self.gpu_block)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))


if __name__ == "__main__":
    unittest.main()
