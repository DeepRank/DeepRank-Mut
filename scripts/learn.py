#!/usr/bin/env python

import sys
import os
import logging

import torch.optim as optim

# Assure that python can find the deeprank files:
deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deeprank_root)

from deeprank.learn.NeuralNet import NeuralNet
from deeprank.learn.DataSet import DataSet
from deeprank.learn.model3d import cnn_class
from deeprank.models.metrics import OutputExporter
from deeprank.models.metrics import TensorboardBinaryClassificationExporter


logging.basicConfig(filename="learn-%d.log" % os.getpid(), filemode="w", level=logging.INFO)


def interpret_args(args, usage):
    """ Convert a list of commandline arguments into a set of positional and keyword arguments.

        Args (list of str): the commandline arguments

        Returns: (tuple(list of str, dict of str)): the positional and keyword arguments
    """

    if len(args) == 0:
        print(usage)
        sys.exit(1)

    if "--help" in args or "-h" in args:
        print(usage)
        sys.exit(0)

    positional_args = []
    kwargs = {}
    i = 0
    while i < len(args):

        if args[i].startswith("--"):
            key = args[i][2:]

            i += 1
            kwargs[key] = args[i]

        elif args[i].startswith("-"):
            key = args[i][1:2]

            if len(args[i]) > 2:
                kwargs[key] = args[i][2:]
            else:
                i += 1
                kwargs[key] = args[i]
        else:
            positional_args.append(args[i])

        i += 1

    return (positional_args, kwargs)


if __name__ == "__main__":

    usage = "Usage: %s [-e EPOCH_COUNT] *preprocessed_hdf5_files" % sys.argv[0]

    args, kwargs = interpret_args(sys.argv[1:], usage)

    dataset = DataSet(args, normalize_features=True)

    if len(args) == 0:
        raise RuntimeError("No preprocessed HDF5 files given")

    epoch_count = int(kwargs.get('e', 50))

    run_directory = "run-{}".format(os.getpid())

    neural_net = NeuralNet(dataset, cnn_class, model_type='3d',task='class', cuda=False,
                           metrics_exporters=[OutputExporter(run_directory),
                                              TensorboardBinaryClassificationExporter(run_directory)])
    neural_net.optimizer = optim.AdamW(neural_net.net.parameters(), lr=0.001, weight_decay=0.005)
    neural_net.train(nepoch = epoch_count, divide_trainset=None, train_batch_size = 5, num_workers=0)
