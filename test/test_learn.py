import os
from tempfile import mkdtemp, mkstemp
from shutil import rmtree

import h5py
import numpy
import torch.optim as optim
from nose.tools import ok_, eq_

from deeprank.models.variant import PdbVariantSelection, VariantClass
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.learn.DataSet import DataSet
from deeprank.learn.NeuralNet import NeuralNet
from deeprank.learn.model3d import cnn_class
from deeprank.models.environment import Environment
from deeprank.domain.amino_acid import *
from deeprank.models.metrics import OutputExporter, TensorboardBinaryClassificationExporter
import deeprank.config


deeprank.config.DEBUG = True


def test_learn():
    """ This test will simply run deeprank's learning code. It doesn't
        test any particular feature or target classes.

        The result of deeprank's learning is not verified. This test
        only runs the code to be sure there are no exceptions thrown.
    """

    feature_modules = ["test.feature.feature1", "test.feature.feature2"]
    target_modules = ["test.target.target1"]

    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
       'number_of_points': [30,30,30],
       'resolution': [1.,1.,1.],
       'atomic_densities': atomic_densities,
    }

    environment = Environment(pdb_root="test/data/pdb", pssm_root="test/data/pssm")

    variants = [PdbVariantSelection("101m", "A", 10, valine, cysteine,
                                    protein_accession="P02144", protein_residue_number=10,
                                    variant_class=VariantClass.BENIGN),
                PdbVariantSelection("101m", "A", 8, glutamine, cysteine,
                                    protein_accession="P02144",
                                    variant_class=VariantClass.PATHOGENIC),
                PdbVariantSelection("101m", "A", 9, glutamine, cysteine,
                                    protein_accession="P02144", protein_residue_number=9,
                                    variant_class=VariantClass.PATHOGENIC)]

    augmentation = 5

    work_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(work_dir_path, "test.hdf5")

        # data_augmentation has been set to a high number, so that
        # the train, valid and test set can be large enough.
        data_generator = DataGenerator(environment, variants,
                                       data_augmentation=augmentation,
                                       compute_targets=target_modules,
                                       compute_features=feature_modules,
                                       hdf5=hdf5_path)

        data_generator.create_database()

        data_generator.map_features(grid_info)

        dataset = DataSet(hdf5_path, grid_info=grid_info,
                          select_feature='all',
                          select_target='target1',
                          normalize_features=False)

        eq_(len(dataset), len(variants) * (augmentation + 1))
        ok_(dataset[0] is not None)

        metrics_directory = os.path.join(work_dir_path, "runs")

        neural_net = NeuralNet(dataset, cnn_class, model_type='3d',task='class',
                               cuda=False, metrics_exporters=[OutputExporter(metrics_directory),
                                                              TensorboardBinaryClassificationExporter(metrics_directory)])

        neural_net.optimizer = optim.SGD(neural_net.net.parameters(),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=0.005)

        neural_net.train(nepoch=10, divide_trainset=0.8, train_batch_size=2, num_workers=0)
    finally:
        rmtree(work_dir_path)
