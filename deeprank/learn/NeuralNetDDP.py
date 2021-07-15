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
import tempfile

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from deeprank.config import logger
from deeprank.learn import DataSet, classMetrics, rankingMetrics
from torch.autograd import Variable 


matplotlib.use('agg')


class NeuralNetDDP():

    def __init__(self, data_set, model,
                 model_type='3d', proj2d=0, task='class',
                 class_weights = None,
                 pretrained_model=None,
                 cuda=False, ngpu=0,
                 plot=True,
                 save_classmetrics=False,
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
            save_classmetrics (bool): Save and plot classification metrics.
                Classification metrics include:
                - accuracy(ACC)
                - sensitivity(TPR)
                - specificity(TNR)
            outdir (str): output directory
        Raises:
            ValueError: if dataset format is not recognized
            ValueError: if task is not recognized
        """
        self.data_set = data_set
        self.pretrained_model = pretrained_model
        self.class_weights = class_weights
        self.cuda = cuda
        self.ngpu = ngpu
        self.task = task
        self.plot = plot
        self.outdir = outdir
        self.save_classmetrics = save_classmetrics
       

        if isinstance(self.data_set, (str, list)) and pretrained_model is None:
            raise ValueError(
                'Argument data_set must be a DeepRankDataSet object\
                              when no pretrained model is loaded')

        if self.pretrained_model is not None:
            if not cuda:
                raise SystemError(
                    'Cuda is necessary for distributed data parallelism.\
                     If cuda is not available please use NeuralNet. (not NeuralNetDPP)\
                     Or add cuda=True to your NeuralNetDDP call.'
                )
            else:
                self.state = torch.load(self.pretrained_model)

            # create the dataset if required
            # but don't process it yet
            if isinstance(self.data_set, (str, list)):
                self.data_set = DataSet(self.data_set, process=False)

            self.load_data_params()
            self.data_set.process_dataset()

        if model_type == '2d':
            self.data_set.transform = True
            self.data_set.proj2D = proj2d
            self.data_set.get_input_shape()


        if self.ngpu > 0:
            self.cuda = True

        if self.ngpu == 0 and self.cuda:
            self.ngpu = 1

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

        if self.save_classmetrics:
            self.metricnames = ['acc', 'tpr', 'tnr']

        #rank is an unique id for each gpu
        rank = int(os.environ["RANK"])
        model = model(self.data_set.input_shape).to(rank)
        ddp_model = DDP(model,device_ids=[rank])
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d'% rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))
        
        self.optimizer = optimizer
        self.net = ddp_model

        if not self.pretrained_model:
            dist.destroy_process_group()


        if self.pretrained_model:
            if self.state['cuda']:
                for paramname in list(self.state['state_dict'].keys()):
                    paramname_new = paramname.lstrip('module.')
                    self.state['state_dict'][paramname_new] = \
                        self.state['state_dict'][paramname]
                    del self.state['state_dict'][paramname]
            self.load_model_params()

        
        if self.pretrained_model:
            self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=0.005,
                                    momentum=0.9,
                                    weight_decay=0.001)
        if self.pretrained_model:
            self.load_optimizer_params()


        self.outdir = outdir
        if self.plot:
            if not os.path.isdir(self.outdir):
                os.mkdir(outdir)


    def train(self,
              nepoch=50,
              divide_trainset=None,
              hdf5='epoch_data.hdf5',
              train_batch_size=10,
              preshuffle=True,
              preshuffle_seed=None,
              export_intermediate=True,
              num_workers=1,
              save_model='best',
              save_epoch='intermediate',
              rank=0):
        """Perform a simple training of the model.
        Args:
            nepoch (int, optional): number of iterations
            divide_trainset (list, optional): the percentage assign to
                the training, validation and test set.
                Examples: [0.7, 0.2, 0.1], [0.8, 0.2], None
            hdf5 (str, optional): file to store the training results
            train_batch_size (int, optional): size of the batch
            preshuffle (bool, optional): preshuffle the dataset before
                dividing it.
            preshuffle_seed (int, optional): set random seed for preshuffle
            export_intermediate (bool, optional): export data at
                intermediate epochs.
            num_workers (int, optional): number of workers to be used to
                prepare the batch data
            save_model (str, optional): 'best' or 'all', save only the
                best model or all models.
            save_epoch (str, optional): 'intermediate' or 'all',
                save the epochs data to HDF5.
        """
        self.rank = rank

        if self.cuda:
            logger.info(f': NGPU      : {self.ngpu}')

        if divide_trainset is not None:
            index_train, index_valid, index_test = self._divide_dataset(
                divide_trainset, preshuffle, preshuffle_seed)
        else:
            index_train = self.data_set.index_train
            index_valid = self.data_set.index_valid
            index_test = self.data_set.index_test

        t0 = time.time()
        self._train(index_train, index_valid, index_test,
                    nepoch=nepoch,
                    train_batch_size=train_batch_size,
                    export_intermediate=export_intermediate,
                    num_workers=num_workers,
                    save_epoch=save_epoch,
                    save_model=save_model)
        self.save_model(filename='last_model.pth.tar')


    def test(self, hdf5='test_data.hdf5'):
        """Test a predefined model on a new dataset.
        Args:
            hdf5 (str, optional): hdf5 file to store the test results
        """
        fname = os.path.join(self.outdir, hdf5)
        self.f5 = h5py.File(fname, 'w')
        self.load_nn_params()

        index = list(range(self.data_set.__len__()))
        sampler = data_utils.sampler.SubsetRandomSampler(index)
        loader = data_utils.DataLoader(self.data_set, sampler=sampler)

        self.data = {}
        _, self.data['test'] = self._epoch("test", loader, train_model=False)
        if self.plot:
            if self.task == 'reg':
                self._plot_scatter_reg(os.path.join(self.outdir, 'prediction.png'))
            else:
                self._plot_boxplot_class(os.path.join(self.outdir, 'prediction.png'))

            self.plot_hit_rate(os.path.join(self.outdir + 'hitrate.png'))

        self._export_epoch_hdf5(0, self.data)
        self.f5.close()

    def __create_state_dict(self):
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


        return state

    def save_model(self, filename='model.pth.tar'):
        """save the model to disk.

        Args:
            filename (str, optional): name of the file
        """
        filename = os.path.join(self.outdir, filename)
        state = self.__create_state_dict()
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

    def __load_data_params(self):
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
        self.data_set.clip_features = self.state['clip_features']
        self.data_set.clip_factor = self.state['clip_factor']
        self.data_set.grid_shape = self.state['grid_shape']
        self.data_set.grid_info = self.state['grid_info']
    
    def __set_cuda(self):
        if self.ngpu > 0:
            self.cuda = True

        if self.ngpu == 0 and self.cuda:
            self.ngpu = 1

        if self.cuda and not torch.cuda.is_available():
            logger.error(
                f' --> CUDA not deteceted: Make sure that CUDA is installed '
                f'and that you are running on GPUs.\n'
                f' --> To turn CUDA of set cuda=False in NeuralNet.\n'
                f' --> Aborting the experiment \n\n')
            sys.exit()


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
        if not isinstance(divide_set, list):
            divide_set = [divide_set, 1. - divide_set]

        if len(divide_set) == 3 and self.data_set.test_database is not None:
            divide_set = [divide_set[0], 1. - divide_set[0]]
            logger.info(f'  : test data set AND test in training set detected\n'
                        f'  : Divide training set as '
                        f'{divide_set[0]} train {divide_set[1]} valid.\n'
                        f'  : Keep test set for testing')

        if preshuffle:
            if preshuffle_seed is not None and not isinstance(
                    preshuffle_seed, int):
                preshuffle_seed = int(preshuffle_seed)
            np.random.seed(preshuffle_seed)
            np.random.shuffle(self.data_set.index_train)

        ntrain = int(np.ceil(float(self.data_set.ntrain) * divide_set[0]))
        nvalid = int(np.floor(float(self.data_set.ntrain) * divide_set[1]))
        index_train = self.data_set.index_train[:ntrain]
        index_valid = self.data_set.index_train[ntrain:ntrain + nvalid]

        if len(divide_set) == 3:
            index_test = self.data_set.index_train[ntrain + nvalid:]
        else:
            index_test = self.data_set.index_test

        return index_train, index_valid, index_test

    def _train(self, index_train, index_valid, index_test,
               nepoch=50, train_batch_size=5,
               export_intermediate=False, num_workers=1,
               save_epoch='intermediate', save_model='best'):
        """Train the model.
        Args:
            index_train (list(int)): Indices of the training set
            index_valid (list(int)): Indices of the validation set
            index_test  (list(int)): Indices of the testing set
            nepoch (int, optional): numbr of epoch
            train_batch_size (int, optional): size of the batch
            export_intermediate (bool, optional):export itnermediate data
            num_workers (int, optional): number of workers pytorch
                uses to create the batch size
            save_epoch (str,optional): 'intermediate' or 'all'
            save_model (str, optional): 'all' or 'best'
        Returns:
            torch.tensor: Parameters of the network after training
        """
        nprint = np.max([1, int(nepoch / 10)])
        
        pin = False
        if self.cuda:
            pin = True

        train_sampler = data_utils.sampler.SubsetRandomSampler(index_train)
        valid_sampler = data_utils.sampler.SubsetRandomSampler(index_valid)
        test_sampler = data_utils.sampler.SubsetRandomSampler(index_test)
        _valid_ = len(valid_sampler.indices) > 0
        _test_ = len(test_sampler.indices) > 0

        self.losses = {'train': []}
        if _valid_:
            self.losses['valid'] = []
        if _test_:
            self.losses['test'] = []

        if self.save_classmetrics:
            self.classmetrics = {}
            for i in self.metricnames:
                self.classmetrics[i] = {'train': []}
                if _valid_:
                    self.classmetrics[i]['valid'] = []
                if _test_:
                    self.classmetrics[i]['test'] = []

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

        min_error = {'train': float('Inf'),
                     'valid': float('Inf'),
                     'test': float('Inf')}

        av_time = 0.0
        self.data = {}
        for epoch in range(nepoch):
            logger.info(f'\n: epoch {epoch:03d} / {nepoch:03d} {"-"*45}')
            t0 = time.time()
            logger.info(f"\n\t=> train the model\n")
            train_loss, self.data['train'] = self._epoch(
                "train-epoch-%d" % epoch,
                train_loader, train_model=True)
            self.losses['train'].append(train_loss)
            if self.save_classmetrics:
                for i in self.metricnames:
                    self.classmetrics[i]['train'].append(self.data['train'][i])

            if _valid_:
                logger.info(f"\n\t=> validate the model\n")
                valid_loss, self.data['valid'] = self._epoch(
                    "valid-epoch-%d" % epoch,
                    valid_loader, train_model=False)
                self.losses['valid'].append(valid_loss)
                if self.save_classmetrics:
                    for i in self.metricnames:
                        self.classmetrics[i]['valid'].append(
                            self.data['valid'][i])
            if _test_:
                logger.info(f"\n\t=> test the model\n")
                test_loss, self.data['test'] = self._epoch(
                    "test-epoch-%d" % epoch,
                    test_loader, train_model=False)
                self.losses['test'].append(test_loss)
                if self.save_classmetrics:
                    for i in self.metricnames:
                        self.classmetrics[i]['test'].append(
                            self.data['test'][i])

            logger.info(f'\n  train loss      : {train_loss:1.3e}')
            if _valid_:
                logger.info(f'  valid loss      : {valid_loss:1.3e}')
            if _test_:
                logger.info(f'  test loss       : {test_loss:1.3e}')
            elapsed = time.time() - t0


            for mode in ['train', 'valid', 'test']:
                if mode not in self.losses:
                    continue
                if self.losses[mode][-1] < min_error[mode]:
                    self.save_model(
                        filename="best_{}_model.pth.tar".format(mode))
                    min_error[mode] = self.losses[mode][-1]

            if save_model == 'all':
                self.save_model(filename="model_epoch_%04d.pth.tar" % epoch)

            if (export_intermediate and epoch % nprint == nprint - 1) or \
                epoch == 0 or epoch == nepoch - 1:

                if self.plot:
                    figname = os.path.join(self.outdir,
                        f"prediction_{epoch:04d}.png")
                    self._plot_scatter(figname)
        

        if self.plot:
            self._export_losses(os.path.join(self.outdir, 'losses-test.png'))
            if self.save_classmetrics:
                for i in self.metricnames:
                    self._export_metrics(i)

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
        epoch_logger = NeuralNetDDP._get_epoch_logger(epoch_name)

        # variables of the epoch
        running_loss = 0
        data = {'outputs': [], 'targets': [], 'mol': []}

        if self.save_classmetrics:
            for i in self.metricnames:
                data[i] = None

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

            fname, molname = mol[0], mol[1]
            data['mol'] += [(f, m) for f, m in zip(fname, molname)]

        # transform the output back
        data['outputs'] = np.array(data['outputs'])  # .flatten()
        data['targets'] = np.array(data['targets'])  # .flatten()

        # make np for export
        data['mol'] = np.array(data['mol'], dtype=object)

        # get classification metrics
        if self.save_classmetrics:
            for i in self.metricnames:
                data[i] = self._get_classmetrics(data, i)

        # normalize the loss
        if n != 0:
            running_loss /= n
        else:
            warnings.warn(f'Empty input in data_loader {data_loader}.')

        epoch_logger.info("running loss: {}".format(running_loss))

        return running_loss, data



    def _get_variables(self, inputs, targets):
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

        inputs, targets = Variable(inputs).float(), Variable(targets).float().to(self.rank)

        if self.task == 'class':
            targets = targets.long()

        return inputs, targets

    def _export_losses(self, figname):
        """Plot the losses vs the epoch.
        Args:
            figname (str): name of the file where to export the figure
        """

        logger.info('\n --> Loss Plot')

        color_plot = ['red', 'blue', 'green']
        labels = ['Train', 'Valid', 'Test']

        fig, ax = plt.subplots()
        for ik, name in enumerate(self.losses):
            plt.plot(np.array(self.losses[name]),
                     c = color_plot[ik],
                     label = labels[ik])

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Losses')

        fig.savefig(figname)
        plt.close()

        fname = os.path.join(self.outdir, "test.hdf5")
        self.f5 = h5py.File(f"{fname}-{self.rank}", 'w')
        grp = self.f5.create_group('/losses/')
        grp.attrs['type'] = 'losses'
        for k, v in self.losses.items():
            grp.create_dataset(k, data=v)    
    
    
    def _export_metrics(self, metricname):
        logger.info(f'\n --> {metricname.upper()} Plot')
        color_plot = ['red', 'blue', 'green']
        labels = ['Train', 'Valid', 'Test']
        data = self.classmetrics[metricname]
        fig, ax = plt.subplots()
        for ik, name in enumerate(data):
            plt.plot(np.array(data[name]), c=color_plot[ik], label=labels[ik])

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metricname.upper())

        figname = os.path.join(self.outdir, metricname + '.png')
        fig.savefig(figname)
        plt.close()

        grp = self.f5.create_group(metricname)
        grp.attrs['type'] = metricname
        for k, v in data.items():
            grp.create_dataset(k, data=v)    


    def _plot_scatter_reg(self, figname):
        """Plot a scatter plots of predictions VS targets.
        Useful to visualize the performance of the training algorithm
        Args:
            figname (str): filename
        """

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

                ax[0, iwin].boxplot(data)
                ax[0, iwin].set_xlabel(l)
                ax[0, iwin].set_xticklabels(['0', '1'])
                iwin += 1
        fig.savefig(figname, bbox_inches='tight')
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
        pred = self._get_binclass_prediction(data)
        targets = data['targets']
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

    def _export_epoch_hdf5(self, epoch, data):
        """Export the epoch data to the hdf5 file.
        Export the data of a given epoch in train/valid/test group.
        In each group are stored the predcited values (outputs),
        ground truth (targets) and molecule name (mol).
        Args:
            epoch (int): index of the epoch
            data (dict): data of the epoch
        """
        hdf5='epoch_data.hdf5'
        fname = os.path.join(self.outdir, hdf5)
        self.f5 = h5py.File(fname, 'w')
        grp_name = 'epoch_%04d' % epoch
        grp = self.f5.create_group(grp_name)
        grp.attrs['type'] = 'epoch'
        grp.attrs['task'] = self.task
        for pass_type, pass_data in data.items():
            try:
                sg = grp.create_group(pass_type)
                for data_name, data_value in pass_data.items():
                    if data_name == 'mol':
                        string_dt = h5py.special_dtype(vlen=str)
                        sg.create_dataset(
                            data_name, data=data_value, dtype=string_dt)
                    else:
                        sg.create_dataset(data_name, data=data_value)
            except TypeError:
                logger.exception("Error in exporting epoch information to hdf5 file")
