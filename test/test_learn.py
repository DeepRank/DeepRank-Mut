import os
from tempfile import mkdtemp, mkstemp
from shutil import rmtree

import h5py
import numpy
import torch.optim as optim
from nose.tools import ok_

from deeprank.models.variant import PdbVariantSelection
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.learn.DataSet import DataSet
from deeprank.learn.NeuralNet import NeuralNet
from deeprank.learn.model3d import cnn_reg
from deeprank.domain.amino_acid import valine, cysteine, serine
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

    variants = [PdbVariantSelection("test/101m.pdb", "A", 10, valine, cysteine, {"A": "test/101M.A.pdb.pssm"},
                                    protein_accession="P02144", protein_residue_number=10),
                PdbVariantSelection("test/data/pdb/5EYU/5EYU.pdb", "A", 8, serine, cysteine, {"A": "test/data/pssm/5EYU/5eyu.A.pdb.pssm",
                                                                                              "B": "test/data/pssm/5EYU/5eyu.B.pdb.pssm",
                                                                                              "C": "test/data/pssm/5EYU/5eyu.C.pdb.pssm",
                                                                                              "D": "test/data/pssm/5EYU/5eyu.D.pdb.pssm"},
                                    protein_accession="Q9L4P8")]

    work_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(work_dir_path, "test.hdf5")

        # data_augmentation has been set to a high number, so that
        # the train, valid and test set can be large enough.
        data_generator = DataGenerator(variants, data_augmentation=25,
                                       compute_targets=target_modules,
                                       compute_features=feature_modules,
                                       hdf5=hdf5_path)

        data_generator.create_database()

        data_generator.map_features(grid_info)

        dataset = DataSet(hdf5_path, grid_info=grid_info,
                          select_feature='all',
                          select_target='target1',
                          normalize_features=False)

        ok_(len(dataset) > 0)
        ok_(dataset[0] is not None)

        net_output_dir_path = os.path.join(work_dir_path, 'net-output')
        neural_net = NeuralNet(dataset, cnn_reg, model_type='3d',task='reg',
                               cuda=False, plot=True, outdir=net_output_dir_path)

        neural_net.optimizer = optim.SGD(neural_net.net.parameters(),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=0.005)

        epoch_data_path = "epoch_data.hdf5"

        neural_net.train(nepoch = 50, divide_trainset=0.8, train_batch_size = 5, num_workers=0, hdf5=epoch_data_path)

        # Check the contents of the variant data output
        with h5py.File(os.path.join(work_dir_path, "net-output", epoch_data_path), 'r') as f5:

            variant_data = f5['epoch_0000/train/variant'][()]
            assert len(variant_data.shape) == 2, "unexpected variant data shape: {}".format(variant_data.shape)
            assert variant_data.shape[1] == 7, "unexpected variant data row format: {}".format(variant_data[0, :])
            assert variant_data[0, 0].decode().lower().endswith(".pdb"), "unexpected structure {}".format(variant_data[0, 0])

    finally:
        rmtree(work_dir_path)


def test_plot_mcc():

    plot_file, plot_path = mkstemp(prefix="plot-mcc", suffix=".png")
    os.close(plot_file)

    try:
        with h5py.File("test/data/epoch_data.hdf5", "r") as f5:
            NeuralNet.plot_mcc(f5, plot_path)
    finally:
        if os.path.isfile(plot_path):
            os.remove(plot_path)
