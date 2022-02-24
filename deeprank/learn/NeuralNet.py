#!/usr/bin/env python
import os
import sys
import time
import logging

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

from deeprank.operate.hdf5data import load_variant
from deeprank.config import logger
from deeprank.learn import DataSet, classMetrics, rankingMetrics
from torch.autograd import Variable
from deeprank.tools.metrics import get_tp_tn_fp_fn, get_mcc
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('agg')


class NeuralNet():

    def __init__(self, data_set, model,
                 model_type='3d', proj2d=0, task='class',
                 class_weights = None,
                 pretrained_model=None,
                 cuda=False, ngpu=0,
                 plot=True,
                 outdir='./'):
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

            plot (bool): Plot the prediction results.

            outdir (str): output directory

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
            self._plot_scatter = self._plot_scatter_reg

        elif self.task == 'class':
            self.criterion = nn.CrossEntropyLoss(weight = self.class_weights, reduction='mean')
            self._plot_scatter = self._plot_boxplot_class
        else:
            raise ValueError(
                f"Task {self.task} not recognized. Options are:\n\t "
                f"reg': regression \n\t 'class': classifiation\n")

        # ------------------------------------------
        # Output
        # ------------------------------------------

        # plot or not plot
        self.plot = plot

        # output directory
        self.outdir = outdir
        if self.plot:
            if not os.path.isdir(self.outdir):
                os.mkdir(outdir)

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
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=0.005,
                                   momentum=0.9,
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
              tensorboard_directory=None,
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

            tensorboard_directory (path, optional): where to store the training results

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
                    tensorboard_directory=tensorboard_directory,
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

    def test(self, tensorboard_directory=None):
        """Test a predefined model on a new dataset.

        Args:
            tensorboard_directory (path, optional): where to store the test results

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
        self.data = {}
        _, self.data['test'] = self._epoch("test", loader, train_model=False)
        if self.task == 'reg':
            self._plot_scatter_reg(os.path.join(self.outdir, 'prediction.png'))
        else:
            self._plot_boxplot_class(os.path.join(self.outdir, 'prediction.png'))

        self.plot_hit_rate(os.path.join(self.outdir + 'hitrate.png'))

        with SummaryWriter(log_dir=tensorboard_directory,
                           comment="test") as tensorboard_writer:

            self._export_epoch_tensorboard(0, "testing", self.task, self.data, tensorboard_writer)

    def save_model(self, filename='model.pth.tar'):
        """save the model to disk.

        Args:
            filename (str, optional): name of the file
        """
        filename = os.path.join(self.outdir, filename)

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
               tensorboard_directory=None,
               num_workers=1,
               save_model='best'):
        """Train the model.

        Args:
            index_train (list(int)): Indices of the training set
            index_valid (list(int)): Indices of the validation set
            index_test  (list(int)): Indices of the testing set
            nepoch (int, optional): numbr of epoch
            train_batch_size (int, optional): size of the batch
            tensorboard_directory (directory path, optional): where to store the results
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

        with SummaryWriter(log_dir=tensorboard_directory,
                           filename_suffix="_train",
                           comment="train") as tensorboard_writer:

            # training loop
            av_time = 0.0
            self.data = {}
            for epoch in range(nepoch):

                logger.info(f'\n: epoch {epoch:03d} / {nepoch:03d} {"-"*45}')
                t0 = time.time()

                # train the model
                logger.info(f"\n\t=> train the model\n")
                train_loss, self.data['train'] = self._epoch("train-epoch-%d" % epoch,
                                                             train_loader, train_model=True)

                self._export_epoch_tensorboard(epoch, "training", self.task, self.data['train'], tensorboard_writer)

                self.losses['train'].append(train_loss)

                # validate the model
                if _valid_:
                    logger.info(f"\n\t=> validate the model\n")
                    valid_loss, self.data['valid'] = self._epoch("valid-epoch-%d" % epoch,
                                                                 valid_loader, train_model=False)

                    self._export_epoch_tensorboard(epoch, "validation", self.task, self.data['valid'], tensorboard_writer)

                    self.losses['valid'].append(valid_loss)

                # test the model
                if _test_:
                    logger.info(f"\n\t=> test the model\n")
                    test_loss, self.data['test'] = self._epoch("test-epoch-%d" % epoch,
                                                               test_loader, train_model=False)

                    self._export_epoch_tensorboard(epoch, "testing", self.task, self.data['test'], tensorboard_writer)

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
                        self.save_model(
                            filename="best_{}_model.pth.tar".format(mode))
                        min_error[mode] = self.losses[mode][-1]

                # save all the model if required
                if save_model == 'all':
                    self.save_model(filename="model_epoch_%04d.pth.tar" % epoch)

                if self.plot:
                    figname = os.path.join(self.outdir,
                        f"prediction_{epoch:04d}.png")
                    self._plot_scatter(figname)

                sys.stdout.flush()

        return torch.cat([param.data.view(-1)
                          for param in self.net.parameters()], 0)


    @staticmethod
    def _get_epoch_logger(epoch_name):
        log_path = "deeprank-%d_%s.log" % (os.getpid(), epoch_name)

        logger = logging.getLogger(epoch_name)

        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

        logger.setLevel(logging.DEBUG)

        return logger

    def _epoch(self, epoch_name, data_loader, train_model):
        """Perform one single epoch iteration over a data loader.

        Args:
            data_loader (torch.DataLoader): DataLoader for the epoch
            train_model (bool): train the model if True or not if False

        Returns:
            float: loss of the model
            dict:  data of the epoch
        """

        # intermediate logger
        epoch_logger = NeuralNet._get_epoch_logger(epoch_name)

        # variables of the epoch
        running_loss = 0
        data = {'outputs': [], 'targets': [], 'mol': [], "variant": []}

        n = 0
        debug_time = False
        time_learn = 0

        # set train/eval mode
        self.net.train(mode=train_model)
        torch.set_grad_enabled(train_model)

        mini_batch = 0

        epoch_logger.info("running epoch on {} data entries".format(len(data_loader)))

        for d in data_loader:

            mini_batch = mini_batch + 1

            logger.info(f"\t\t-> mini-batch: {mini_batch} ")

            # get the data
            inputs = d['feature']
            targets = d['target']
            mol = d['mol']

            epoch_logger.debug("data entry {}".format(mol))
            for input_index, input_ in enumerate(inputs):
                input_ = np.array(input_)
                input_summary = "%s<{%f - %f}" % ("x".join([str(n) for n in input_.shape]), np.min(input_), np.max(input_))
                epoch_logger.debug("  has input {}: {}\n{}".format(input_index, input_summary, input_))
            epoch_logger.debug("  has target: {}".format(targets))

            # transform the data
            inputs, targets = self._get_variables(inputs, targets)

            # starting time
            tlearn0 = time.time()

            # forward
            outputs = self.net(inputs)

            epoch_logger.debug("data entry {}:\n  has output:{}".format(mol, outputs))

            # class complains about the shape ...
            if self.task == 'class':
                targets = targets.view(-1)

            # evaluate loss
            loss = self.criterion(outputs, targets)

            epoch_logger.debug("data entry {}:\n  has loss:{}".format(mol, loss))

            running_loss += loss.data.item()  # pytorch1 compatible
            n += len(inputs)

            # zero + backward + step
            if train_model:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            time_learn += time.time() - tlearn0

            # get the outputs for export
            if self.cuda:
                data['outputs'] += outputs.data.cpu().numpy().tolist()
                data['targets'] += targets.data.cpu().numpy().tolist()
            else:
                data['outputs'] += outputs.data.numpy().tolist()
                data['targets'] += targets.data.numpy().tolist()

            # write the names of the variant to the epoch data
            fname, molname = mol[0], mol[1]
            for f, m in zip(fname, molname):
                data['mol'] += [(f, m)]

                with h5py.File(f, 'r') as f5:
                    variant = load_variant(f5[m])

                variant_row = [variant.pdb_ac, variant.chain_id, variant.residue_id,
                               variant.wild_type_amino_acid.name, variant.variant_amino_acid.name]

                if variant.protein_accession is not None:
                    variant_row.append(variant.protein_accession)
                else:
                    variant_row.append("")

                if variant.protein_residue_number is not None:
                    variant_row.append(str(variant.protein_residue_number))
                else:
                    variant_row.append("")

                data['variant'].append(variant_row)

        # transform the output back
        data['outputs'] = np.array(data['outputs'])  # .flatten()
        data['targets'] = np.array(data['targets'])  # .flatten()

        # make np for export
        data['mol'] = np.array(data['mol'], dtype=object)
        data['variant'] = np.array(data['variant'], dtype=object)

        # normalize the loss
        if n != 0:
            running_loss /= n
        else:
            warnings.warn(f'Empty input in data_loader {data_loader}.')

        epoch_logger.info("running loss: {}".format(running_loss))
        data['loss'] = running_loss

        return running_loss, data

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

    def _plot_scatter_reg(self, figname):
        """Plot a scatter plots of predictions VS targets.

        Useful to visualize the performance of the training algorithm

        Args:
            figname (str): filename
        """

        # abort if we don't want to plot
        if self.plot is False:
            return

        logger.info(f'\n  --> Scatter Plot: {figname}')

        color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        fig, ax = plt.subplots()

        xvalues = np.array([])
        yvalues = np.array([])

        for l in labels:

            if l in self.data:

                targ = self.data[l]['targets'].flatten()
                out = self.data[l]['outputs'].flatten()

                xvalues = np.append(xvalues, targ)
                yvalues = np.append(yvalues, out)

                ax.scatter(targ, out, c=color_plot[l], label=l)

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')

        values = np.append(xvalues, yvalues)

        border = 0.1 * (values.max() - values.min())
        ax.plot([values.min() - border, values.max() + border],
                [values.min() - border, values.max() + border])

        fig.savefig(figname)
        plt.close()

    def _plot_boxplot_class(self, figname):
        """Plot a boxplot of predictions VS targets.

        It is only usefull in classification tasks.

        Args:
            figname (str): filename
        """

        # abort if we don't want to plot
        if not self.plot:
            return

        logger.info(f'\n  --> Box Plot: {figname}')

        color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        nwin = len(self.data)

        fig, ax = plt.subplots(1, nwin, sharey=True, squeeze=False)

        iwin = 0
        for l in labels:

            if l in self.data:

                tar = self.data[l]['targets']
                out = self.data[l]['outputs']

                data = [[], []]
                confusion = [[0, 0], [0, 0]]
                for pts, t in zip(out, tar):
                    r = F.softmax(torch.FloatTensor(pts), dim=0).data.numpy()
                    data[t].append(r[1])
                    confusion[t][bool(r[1] > 0.5)] += 1

                #print("  {:5s}: {:s}".format(l,str(confusion)))

                ax[0, iwin].boxplot(data)
                ax[0, iwin].set_xlabel(l)
                ax[0, iwin].set_xticklabels(['0', '1'])
                iwin += 1

        fig.savefig(figname, bbox_inches='tight')
        plt.close()


    @staticmethod
    def plot_mcc(data_hdf5_file, figure_path):
        """ Plot MCC values on the Y-axis and epochs on the X-axis

        Args:
            figname (str): filename
        """

        plot_colors = {'train': 'red', 'valid': 'blue', 'test': 'green'}

        # gather data from hdf5 file
        plot_data = {}
        for key in data_hdf5_file:
            if key.startswith("epoch_"):
                epoch_number = int(key.split('_')[1])

                for phase_name in plot_colors:
                    if phase_name in data_hdf5_file[key]:

                        if phase_name not in plot_data:
                            plot_data[phase_name] = {'epochs':[], 'mcc': []}

                        phase_group = data_hdf5_file["{}/{}".format(key, phase_name)]

                        outputs = phase_group["outputs"][()]
                        targets = phase_group["targets"][()]

                        tp, tn, fp, fn = get_tp_tn_fp_fn(outputs, targets)
                        mcc = get_mcc(tp, tn, fp, fn)

                        plot_data[phase_name]['epochs'].append(epoch_number)
                        plot_data[phase_name]['mcc'].append(mcc)

        # create plot
        figure, axis = plt.subplots()
        for phase_name, color in plot_colors.items():
            if phase_name in plot_data:
                plt.plot(plot_data[phase_name]['epochs'], plot_data[phase_name]['mcc'], c=color, label=phase_name)

        legend = axis.legend(loc='upper left')
        axis.set_xlabel("epoch")
        axis.set_ylabel("MCC")

        figure.savefig(figure_path)
        plt.close()


    def plot_hit_rate(self, figname):
        """Plot the hit rate of the different training/valid/test sets.

        The hit rate is defined as:
            The percentage of positive(near-native) decoys that are
            included among the top m decoys.

        Args:
            figname (str): filename for the plot
            irmsd_thr (float, optional): threshold for 'good' models
        """
        if self.plot is False:
            return

        logger.info(f'\n  --> Hitrate plot: {figname}\n')

        color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        fig, ax = plt.subplots()
        for l in labels:
            if l in self.data:
                if 'hit' in self.data[l]:
                    hitrate = rankingMetrics.hitrate(self.data[l]['hit'])
                    m = len(hitrate)
                    x = np.linspace(0, 100, m)
                    plt.plot(x, hitrate, c=color_plot[l], label=f"{l} M={m}")
        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Top M (%)')
        ax.set_ylabel('Hit Rate')

        fmt = '%.0f%%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks)

        fig.savefig(figname)
        plt.close()

    def _get_classmetrics(self, data, metricname):

        # get predctions
        pred = self._get_binclass_prediction(data)

        # get real targets
        targets = data['targets']

        # get metric values
        if metricname == 'acc':
            return classMetrics.accuracy(pred, targets)
        elif metricname == 'tpr':
            return classMetrics.sensitivity(pred, targets)
        elif metricname == 'tnr':
            return classMetrics.specificity(pred, targets)
        elif metricname == 'ppv':
            return classMetrics.precision(pred, targets)
        elif metricname == 'f1':
            return classMetrics.F1(pred, targets)
        else:
            return None

    @staticmethod
    def _get_binclass_prediction(data):

        out = data['outputs']
        probility = F.softmax(torch.FloatTensor(out), dim=1).data.numpy()
        pred = probility[:, 0] <= probility[:, 1]
        return pred.astype(int)

    @staticmethod
    def _export_epoch_tensorboard(epoch_number, pass_, task, epoch_data, tensorboard_writer):

        if 'loss' in epoch_data:
            loss = epoch_data['loss']
            tensorboard_writer.add_scalar("loss", loss, epoch_number)

        if task == "class":
            tp, tn, fp, fn = get_tp_tn_fp_fn(epoch_data['outputs'], epoch_data['targets'])

            mcc = get_mcc(tp, tn, fp, fn)
            tensorboard_writer.add_scalar("MCC", mcc, epoch_number)

            accuracy = (tp + tn) / (tp + tn + fp + fn) 
            tensorboard_writer.add_scalar("accuracy", accuracy, epoch_number)
