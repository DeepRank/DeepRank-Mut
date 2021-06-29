import sys
import os
import logging

import torch.optim as optim

# Assure that python can find the deeprank files:
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deeprank.learn.NeuralNet import NeuralNet
from deeprank.learn.DataSet import DataSet
from deeprank.learn.model3d import cnn_class


logging.basicConfig(filename="learn-%d.log" % os.getpid(), filemode="w", level=logging.INFO)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        raise RuntimeError("at least one hdf5 file input argument is required")

    grid_info = {'number_of_points': [30,30,30], 'resolution': [1.,1.,1.], 'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}}
    dataset = DataSet(sys.argv[1:], grid_info=grid_info, normalize_features=True)

    neural_net = NeuralNet(dataset, cnn_class, model_type='3d',task='class', cuda=False, plot=True, outdir="net-output")
    neural_net.optimizer = optim.SGD(neural_net.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    neural_net.train(nepoch = 50, divide_trainset=[0.7, 0.2, 0.1], train_batch_size = 5, num_workers=0)
