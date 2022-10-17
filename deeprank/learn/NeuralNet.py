#!/usr/bin/env python
import os
import sys
import time
import logging
import csv
from typing import List

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import warnings

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchsummary import summary

from deeprank.operate.hdf5data import load_variant, get_variant_group_name
from deeprank.config import logger
from deeprank.learn import DataSet, classMetrics, rankingMetrics
from torch.autograd import Variable
from deeprank.models.metrics import MetricsExporterCollection, MetricsExporter

matplotlib.use('agg')


class NeuralNet():

    def __init__(self, data_set, model,
                 model_type='3d', proj2d=0, task='class',
                 class_weights = None,
                 pretrained_model=None,
                 cuda=False, ngpu=0,
                 metrics_exporters: List[MetricsExporter] = None):
        """Train a Convolutional Neural Network for DeepRank.

        Args:
            data_set (deeprank.DataSet or list(str)): Data set used for
                training or testing.
                - deeprank.DataSet for training;
                - str or list(str), e.g. 'x.hdf5', ['x1.hdf5', 'x2.hdf5'],
                for testing when pretrained model is loaded.

            model (nn.Module): Definition of the NN to use.
                Must subclass nn.Module.
                See examples in model2d.py and model3d.py

            model_type (str): Type of model we want to use.
                Must be '2d' or '3d'.
                If we specify a 2d model, the data set is automatically
                converted to the correct format.

            proj2d (int): Defines how to slice the 3D volumetric data to generate
                2D data. Allowed values are 0, 1 and 2, which are to slice along
                the YZ, XZ or XY plane, respectively.

            task (str 'reg' or 'class'): Task to perform.
                - 'reg' for regression
                - 'class' for classification.
                The loss function, the target datatype and plot functions
                will be autmatically adjusted depending on the task.

            class_weights (Tensor): a manual rescaling weight given to
                each class. If given, has to be a Tensor of size #classes.
                Only applicable on 'class' task.

            pretrained_model (str): Saved model to be used for further
                training or testing.

            cuda (bool): Use CUDA.

            ngpu (int): number of GPU to be used.

            metrics_exporters: to be used for output during the run

        Raises:
            ValueError: if dataset format is not recognized
            ValueError: if task is not recognized

        Examples:
            Train models:
            >>> data_set = Dataset(...)
            >>> model = NeuralNet(data_set, cnn,
            ...                   model_type='3d', task='reg',
            ...                   plot=True,
            ...                   outdir='./out/')
            >>> model.train(nepoch = 50, divide_trainset=0.8,
            ...             train_batch_size = 5, num_workers=0)

            Test a model on new data:
            >>> data_set = ['test01.hdf5', 'test02.hdf5']
            >>> model = NeuralNet(data_set, cnn,
            ...                   pretrained_model = './model.pth.tar',
            ...                   outdir='./out/')
            >>> model.test()
        """

        # ------------------------------------------
        # Dataset
        # ------------------------------------------

        # data set and model
        self.data_set = data_set

        # pretrained model
        self.pretrained_model = pretrained_model

        self.class_weights = class_weights

        if isinstance(self.data_set, (str, list)) and pretrained_model is None:
            raise ValueError(
                'Argument data_set must be a DeepRankDataSet object\
                              when no pretrained model is loaded')

        # load the model
        if self.pretrained_model is not None:

            if not cuda:
                self.state = torch.load(self.pretrained_model,
                                        map_location='cpu')
            else:
                self.state = torch.load(self.pretrained_model)

            # create the dataset if required
            # but don't process it yet
            if isinstance(self.data_set, (str, list)):
                self.data_set = DataSet(self.data_set, process=False)

            # load the model and
            # change dataset parameters
            self.load_data_params()

            # process it
            self.data_set.process_dataset()

        # convert the data to 2d if necessary
        if model_type == '2d':

            self.data_set.transform = True
            self.data_set.proj2D = proj2d
            self.data_set.get_input_shape()

        # ------------------------------------------
        # CUDA
        # ------------------------------------------

        # CUDA required
        self.cuda = cuda
        self.ngpu = ngpu

        # handles GPU/CUDA
        if self.ngpu > 0:
            self.cuda = True

        if self.ngpu == 0 and self.cuda:
            self.ngpu = 1

        # ------------------------------------------
        # Regression or classifiation
        # ------------------------------------------

        # task to accomplish
        self.task = task

        # Set the loss functiom
        if self.task == 'reg':
            self.criterion = nn.MSELoss(reduction='sum')

        elif self.task == 'class':
            self.criterion = nn.CrossEntropyLoss(weight = self.class_weights, reduction='mean')
        else:
            raise ValueError(
                f"Task {self.task} not recognized. Options are:\n\t "
                f"reg': regression \n\t 'class': classifiation\n")

        # ------------------------------------------
        # Output
        # ------------------------------------------
        self._metrics_output = MetricsExporterCollection(*metrics_exporters)

        # ------------------------------------------
        # Network
        # ------------------------------------------

        # load the model
        self.net = model(self.data_set.input_shape)

        # print model summary
        sys.stdout.flush()
        if cuda is True:
            device = torch.device("cuda")  # PyTorch v0.4.0
        else:
            device = torch.device("cpu")
        summary(self.net.to(device),
                self.data_set.input_shape,
                device=device.type)
        sys.stdout.flush()

        # load parameters of pretrained model if provided
        if self.pretrained_model:
            # a prefix 'module.' is added to parameter names if
            # torch.nn.DataParallel was used
            # https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
            if self.state['cuda']:
                for paramname in list(self.state['state_dict'].keys()):
                    paramname_new = paramname.lstrip('module.')
                    self.state['state_dict'][paramname_new] = \
                        self.state['state_dict'][paramname]
                    del self.state['state_dict'][paramname]
            self.load_model_params()

        # multi-gpu
        if self.ngpu > 1:
            ids = [i for i in range(self.ngpu)]
            self.net = nn.DataParallel(self.net, device_ids=ids).cuda()
        # cuda compatible
        elif self.cuda:
            self.net = self.net.cuda()

        # set the optimizer
        #self.optimizer = optim.SGD(self.net.parameters(),
        #                           lr=0.005,
        #                           momentum=0.9,
        #                           weight_decay=0.001)
        self.optimizer = optim.AdamW(self.net.parameters(),
                                     lr=0.005,
                                     weight_decay=0.001)
        if self.pretrained_model:
            self.load_optimizer_params()

        # ------------------------------------------
        # print
        # ------------------------------------------

        logger.info('\n')
        logger.info('=' * 40)
        logger.info('=\t Convolution Neural Network')
        logger.info(f'=\t model   : {model_type}')
        logger.info(f'=\t CNN      : {model.__name__}')

        for feat_type, feat_names in self.data_set.select_feature.items():
            logger.info(f'=\t features : {feat_type}')
            for name in feat_names:
                logger.info(f'=\t\t     {name}')
        logger.info(f'=\t targets  : {self.data_set.select_target}')
        logger.info(f'=\t CUDA     : {str(self.cuda)}')
        if self.cuda:
            logger.info(f'=\t nGPU     : {self.ngpu}')
        logger.info('=' * 40 + '\n')

        # check if CUDA works
        if self.cuda and not torch.cuda.is_available():
            logger.error(
                f' --> CUDA not deteceted: Make sure that CUDA is installed '
                f'and that you are running on GPUs.\n'
                f' --> To turn CUDA of set cuda=False in NeuralNet.\n'
                f' --> Aborting the experiment \n\n')
            sys.exit()

    def train(self,
              nepoch=50,
              divide_trainset=None,
              train_batch_size=10,
              preshuffle=True,
              preshuffle_seed=None,
              num_workers=1,
              save_model='best'):

        """Perform a simple training of the model.

        Args:
            nepoch (int, optional): number of iterations

            divide_trainset (list, optional): the percentage assign to
                the training, validation and test set.
                Examples: [0.7, 0.2, 0.1], [0.8, 0.2], None

            train_batch_size (int, optional): size of the batch

            preshuffle (bool, optional): preshuffle the dataset before
                dividing it.

            preshuffle_seed (int, optional): set random seed for preshuffle

            num_workers (int, optional): number of workers to be used to
                prepare the batch data

            save_model (str, optional): 'best' or 'all', save only the
                best model or all models.
        """
        logger.info(f'\n: Batch Size: {train_batch_size}')
        if self.cuda:
            logger.info(f': NGPU      : {self.ngpu}')

        # divide the set in train+ valid and test
        if divide_trainset is not None:
            # if divide_trainset is not None
            index_train, index_valid, index_test = self._divide_dataset(
                divide_trainset, preshuffle, preshuffle_seed)
        else:
            index_train = self.data_set.index_train
            index_valid = self.data_set.index_valid
            index_test = self.data_set.index_test

        logger.info(f': {len(index_train)} confs. for training')
        logger.info(f': {len(index_valid)} confs. for validation')
        logger.info(f': {len(index_test)} confs. for testing')

        # train the model
        t0 = time.time()
        self._train(index_train, index_valid, index_test,
                    nepoch=nepoch,
                    train_batch_size=train_batch_size,
                    num_workers=num_workers,
                    save_model=save_model)

        logger.info(
            f' --> Training done in {self.convertSeconds2Days(time.time()-t0)}')

        # save the model
        self.save_model(filename='last_model.pth.tar')

    @staticmethod
    def convertSeconds2Days(time):
        # input time in seconds

        time = int(time)
        day = time // (24 * 3600)
        time = time % (24 * 3600)
        hour = time // 3600
        time %= 3600
        minutes = time // 60
        time %= 60
        seconds = time
        return '%02d-%02d:%02d:%02d' % (day, hour, minutes, seconds)

    def test(self):
        """Test a predefined model on a new dataset.

        Examples:
            >>> # adress of the database
            >>> database = '1ak4.hdf5'
            >>> # Load the model in a new network instance
            >>> model = NeuralNet(database, cnn,
            ...                   pretrained_model='./model/model.pth.tar',
            ...                   outdir='./test/')
            >>> # test the model
            >>> model.test()
        """

        # load pretrained model to get task and criterion
        self.load_nn_params()

        # load data
        index = list(range(self.data_set.__len__()))
        sampler = data_utils.sampler.SubsetRandomSampler(index)
        loader = data_utils.DataLoader(self.data_set, sampler=sampler)

        # do test
        with self._metrics_output:

            self._epoch(0, "test", loader, False)

    def save_model(self, filename='model.pth.tar'):
        """save the model to disk.

        Args:
            filename (str, optional): name of the file
        """
        state = {'state_dict': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'normalize_features': self.data_set.normalize_features,
                 'select_feature': self.data_set.select_feature,
                 'select_target': self.data_set.select_target,
                 'dict_filter': self.data_set.dict_filter,
                 'transform': self.data_set.transform,
                 'proj2D': self.data_set.proj2D,
                 'clip_features': self.data_set.clip_features,
                 'clip_factor': self.data_set.clip_factor,
                 'grid_shape': self.data_set.grid_shape,
                 'grid_info': self.data_set.grid_info,
                 'task': self.task,
                 'criterion': self.criterion,
                 'cuda': self.cuda
                 }

        if self.data_set.normalize_features:
            state['feature_mean'] = self.data_set.feature_mean
            state['feature_std'] = self.data_set.feature_std

        torch.save(state, filename)

    def load_model_params(self):
        """Get model parameters from a saved model."""
        self.net.load_state_dict(self.state['state_dict'])

    def load_optimizer_params(self):
        """Get optimizer parameters from a saved model."""
        self.optimizer.load_state_dict(self.state['optimizer'])

    def load_nn_params(self):
        """Get NeuralNet parameters from a saved model."""
        self.task = self.state['task']
        self.criterion = self.state['criterion']

    def load_data_params(self):
        """Get dataset parameters from a saved model."""
        self.data_set.select_feature = self.state['select_feature']
        self.data_set.select_target = self.state['select_target']

        self.data_set.dict_filter = self.state['dict_filter']

        self.data_set.normalize_features = self.state['normalize_features']
        if self.data_set.normalize_features:
            self.data_set.feature_mean = self.state['feature_mean']
            self.data_set.feature_std = self.state['feature_std']

        self.data_set.transform = self.state['transform']
        self.data_set.proj2D = self.state['proj2D']
        #self.data_set.target_ordering = self.state['target_ordering']
        self.data_set.clip_features = self.state['clip_features']
        self.data_set.clip_factor = self.state['clip_factor']
        self.data_set.grid_shape = self.state['grid_shape']
        self.data_set.grid_info = self.state['grid_info']

    def _divide_dataset(self, divide_set, preshuffle, preshuffle_seed):
        """Divide the data set into training, validation and test
        according to the percentage in divide_set.

        Args:
            divide_set (list(float)): percentage used for
                training/validation/test.
                Example: [0.8, 0.1, 0.1], [0.8, 0.2]
            preshuffle (bool): shuffle the dataset before dividing it
            preshuffle_seed (int, optional): set random seed

        Returns:
            list(int),list(int),list(int): Indices of the
                training/validation/test set.
        """
        # if user only provided one number
        # we assume it's the training percentage
        if not isinstance(divide_set, list):
            divide_set = [divide_set, 1. - divide_set]

        # if user provided 3 number and testset
        if len(divide_set) == 3 and self.data_set.test_database is not None:
            divide_set = [divide_set[0], 1. - divide_set[0]]
            logger.info(f'  : test data set AND test in training set detected\n'
                        f'  : Divide training set as '
                        f'{divide_set[0]} train {divide_set[1]} valid.\n'
                        f'  : Keep test set for testing')

        # preshuffle
        if preshuffle:
            if preshuffle_seed is not None and not isinstance(
                    preshuffle_seed, int):
                preshuffle_seed = int(preshuffle_seed)
            np.random.seed(preshuffle_seed)
            np.random.shuffle(self.data_set.index_train)

        # size of the subset for training
        ntrain = int(np.ceil(float(self.data_set.ntrain) * divide_set[0]))
        nvalid = int(np.floor(float(self.data_set.ntrain) * divide_set[1]))

        # indexes train and valid
        index_train = self.data_set.index_train[:ntrain]
        index_valid = self.data_set.index_train[ntrain:ntrain + nvalid]

        # index of test depending of the situation
        if len(divide_set) == 3:
            index_test = self.data_set.index_train[ntrain + nvalid:]
        else:
            index_test = self.data_set.index_test

        return index_train, index_valid, index_test

    def _train(self, index_train, index_valid, index_test,
               nepoch=50, train_batch_size=5,
               num_workers=1,
               save_model='best'):
        """Train the model.

        Args:
            index_train (list(int)): Indices of the training set
            index_valid (list(int)): Indices of the validation set
            index_test  (list(int)): Indices of the testing set
            nepoch (int, optional): numbr of epoch
            train_batch_size (int, optional): size of the batch
            num_workers (int, optional): number of workers pytorch
                uses to create the batch size
            save_model (str, optional): 'all' or 'best'

        Returns:
            torch.tensor: Parameters of the network after training
        """

        # printing options
        nprint = np.max([1, int(nepoch / 10)])

        # pin memory for cuda
        pin = False
        if self.cuda:
            pin = True

        # create the sampler
        train_sampler = data_utils.sampler.SubsetRandomSampler(index_train)
        valid_sampler = data_utils.sampler.SubsetRandomSampler(index_valid)
        test_sampler = data_utils.sampler.SubsetRandomSampler(index_test)

        # get if we test as well
        _valid_ = len(valid_sampler.indices) > 0
        _test_ = len(test_sampler.indices) > 0

        # containers for the losses
        self.losses = {'train': []}
        if _valid_:
            self.losses['valid'] = []
        if _test_:
            self.losses['test'] = []

        #  create the loaders
        train_loader = data_utils.DataLoader(
            self.data_set,
            batch_size=train_batch_size,
            sampler=train_sampler,
            pin_memory=pin,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True)
        if _valid_:
            valid_loader = data_utils.DataLoader(
                self.data_set,
                batch_size=train_batch_size,
                sampler=valid_sampler,
                pin_memory=pin,
                num_workers=num_workers,
                shuffle=False,
                drop_last=True)
        if _test_:
            test_loader = data_utils.DataLoader(
                self.data_set,
                batch_size=train_batch_size,
                sampler=test_sampler,
                pin_memory=pin,
                num_workers=num_workers,
                shuffle=False,
                drop_last=True)

        # min error to kee ptrack of the best model.
        min_error = {'train': float('Inf'),
                     'valid': float('Inf'),
                     'test': float('Inf')}

        with self._metrics_output:

            # Measure the values at epoch zero:
            self._epoch(0, "training", train_loader, False)
            if _valid_:
                self._epoch(0, "validation", valid_loader, False)
            if _test_:
                self._epoch(0, "testing", test_loader, False)

            # training loop
            av_time = 0.0
            self.data = {}
            for epoch in range(1, nepoch + 1):

                logger.info(f'\n: epoch {epoch:03d} / {nepoch:03d} {"-"*45}')
                t0 = time.time()

                # train the model
                logger.info(f"\n\t=> train the model\n")
                train_loss = self._epoch(epoch, "training", train_loader, True)
                self.losses['train'].append(train_loss)

                # validate the model
                if _valid_:
                    logger.info(f"\n\t=> validate the model\n")
                    valid_loss = self._epoch(epoch, "validation", valid_loader, False)
                    self.losses['valid'].append(valid_loss)

                # test the model
                if _test_:
                    logger.info(f"\n\t=> test the model\n")
                    test_loss = self._epoch(epoch, "testing", test_loader, False)
                    self.losses['test'].append(test_loss)

                # talk a bit about losse
                logger.info(f'\n  train loss      : {train_loss:1.3e}')
                if _valid_:
                    logger.info(f'  valid loss      : {valid_loss:1.3e}')
                if _test_:
                    logger.info(f'  test loss       : {test_loss:1.3e}')

                # timer
                elapsed = time.time() - t0
                logger.info(
                    f'  epoch done in   : {self.convertSeconds2Days(elapsed)}')

                # remaining time
                av_time += elapsed
                nremain = nepoch - (epoch + 1)
                remaining_time = av_time / (epoch + 1) * nremain
                logger.info(f"  remaining time  : "
                    f"{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

                # save the best model
                for mode in ['train', 'valid', 'test']:
                    if mode not in self.losses:
                        continue
                    if self.losses[mode][-1] < min_error[mode]:
                        self.save_model(filename="best_{}_model.pth.tar".format(mode))
                        min_error[mode] = self.losses[mode][-1]

                # save all the model if required
                if save_model == 'all':
                    self.save_model(filename="model_epoch_%04d.pth.tar" % epoch)

    @staticmethod
    def _read_entry_names(batch):
        entry_names = []
        for file_path, group_name in zip(batch['mol'][0], batch['mol'][1]):
            with h5py.File(file_path, 'r') as f5:
                entry_group = f5[group_name]
                variant = load_variant(entry_group)
                entry_name = get_variant_group_name(variant)

                entry_names.append(entry_name)

        return entry_names

    def _epoch(self, epoch_number, pass_name, data_loader, train_model):
        """Perform one single epoch iteration over a data loader.

        Args:
            epoch_index (int): index of the epoch
            pass_name (str): a name for the pass, like: training, validation, testing
            data_loader (torch.DataLoader): DataLoader for the epoch
            train_model (bool): train the model if True or not if False

        Returns:
            float: loss of the model
            dict:  data of the epoch
        """

        sum_of_losses = 0.0
        count_data_entries = 0
        debug_time = False
        time_learn = 0

        # set train/eval mode
        self.net.train(mode=train_model)
        torch.set_grad_enabled(train_model)

        mini_batch = 0

        batch_count = len(data_loader)
        logger.info("running epoch {} on {} batches".format(epoch_number, batch_count))

        entry_names = []
        output_values = []
        output_probs = []
        target_values = []

        for batch_index, batch in enumerate(data_loader):

            mini_batch = mini_batch + 1

            logger.info(f"\t\t-> mini-batch: {mini_batch} ")

            # get the data
            inputs = batch['feature']
            targets = batch['target']
            entry_names += self._read_entry_names(batch)

            # transform the data
            inputs, targets = self._get_variables(inputs, targets)

            # starting time
            tlearn0 = time.time()

            # forward
            outputs = self.net(inputs)

            # class complains about the shape ...
            if self.task == 'class':
                targets = targets.view(-1)

            # evaluate loss
            batch_loss = self.criterion(outputs, targets)

            sum_of_losses += batch_loss.detach().item() * outputs.shape[0]
            count_data_entries += outputs.shape[0]

            # zero + backward + step
            if train_model:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            time_learn += time.time() - tlearn0

            output_values += outputs.tolist()
            target_values += targets.tolist()
            
            output_probs = F.softmax(torch.FloatTensor(output_values), dim=0).tolist()

        self._metrics_output.process(pass_name, epoch_number, entry_names, output_probs, target_values)

        if count_data_entries > 0:
            epoch_loss = sum_of_losses / count_data_entries
        else:
            epoch_loss = 0.0

        return epoch_loss

    def _get_variables(self, inputs, targets):
        # xue: why not put this step to DataSet.py?
        """Convert the feature/target in torch.Variables.

        The format is different for regression where the targets are float
        and classification where they are int.

        Args:
            inputs (np.array): raw features
            targets (np.array): raw target values

        Returns:
            torch.Variable: features
            torch.Variable: target values
        """

        # if cuda is available
        if self.cuda:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # get the varialbe as float by default
        inputs, targets = Variable(inputs).float(), Variable(targets).float()

        # change the targets to long for classification
        if self.task == 'class':
            targets = targets.long()

        return inputs, targets

