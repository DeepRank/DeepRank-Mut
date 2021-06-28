import glob
import os
import pickle
import re
import sys
from functools import partial
import warnings

import h5py
import numpy as np
from tqdm import tqdm
import pdb2sql

from deeprank import config
from deeprank.config import logger
from deeprank.generate import MinMaxParam, NormalizeData, NormParam
from deeprank.tools import sparse

# import torch.utils.data as data_utils
# The class used to subclass data_utils.Dataset
# but that conflict with Sphinx that couldn't build the API
# It's apparently not necessary though and works without subclassing


class DataSet():

    def __init__(self, train_database, valid_database=None, test_database=None,
                 grid_info=None,
                 use_rotation=None,
                 select_feature='all', select_target='class',
                 normalize_features=True,
                 dict_filter=None,
                 transform_to_2D=False, projection=0,
                 grid_shape=None,
                 clip_features=False, clip_factor=1.5,
                 rotation_seed=None,
                 tqdm=False,
                 process=True):
        '''Generates the dataset needed for pytorch.

        This class handles the data generated by deeprank.generate to be
        used in the deep learning part of DeepRank.

        Args:
            train_database (list(str)): names of the hdf5 files used for
                the training/validation.
                Example: ['1AK4.hdf5','1B7W.hdf5',...]
            valid_database (list(str)): names of the hdf5 files used for
                the validation.
                Example: ['1ACB.hdf5','4JHF.hdf5',...]
            test_database (list(str)): names of the hdf5 files used for
                the test.
                Example: ['7CEI.hdf5']

            grid_info(dict): grid information to map the feature on the
                fly. if None the original grid points are used.
                Example:
                    {'number_of_points': [X,Y,Z], 'resolution': [X,Y,Z]}

            use_rotation (int): number of rotations to use.
                Example: 0 (use only original data)
                Default: None  (use all data of the database)

            select_feature (dict or 'all', optional):
                    Select the features used in the learning.
                    - {'_ind': 'all', 'Feature_ind': 'all'}
                    Default: 'all'
            select_target (str): Specify required target.
                Default: 'class' which holds binary values 
                    - 0.0 for benign 
                    - 1.0 for pathogenic 

            normalize_features (Bool, optional): normalize features or not
                Default: True
            dict_filter (None or dict, optional): Specify if we filter
                the variants based on target values (in case of multiple?),
                Example: {'class': '==0.0'}
                (select benign variants)
                Default: None
            transform_to_2D (bool, optional):
                Boolean to use 2d maps instead of full 3d
                Default: False
            projection (int): Projection axis from 3D to 2D:
                Mapping: 0 -> yz, 1 -> xz, 2 -> xy
                Default = 0
            grid_shape (None or tuple(int), optional):
                Shape of the grid in the hdf5 file. Is not necessary
                if the grid points are still present in the HDF5 file.
            clip_features (bool):
                Remove too large values of the grid.
                Can be needed for native complexes where the coulomb
                feature might be too large
                Default: False
            clip_factor (float, optional): the features are clipped at:
                +/-mean + clip_factor * std
            tqdm (bool, optional): Print the progress bar
            process (bool, optional): Actually process the data set.
                Must be set to False when reusing a model for testing
            rotation_seed(int, optional): random seed for getting rotation
                axis and angle.

        Examples:
            >>> from deeprank.learn import *
            >>> train_database = '1ak4.hdf5'
            >>> data_set = DataSet(train_database,
            >>>                    valid_database = None,
            >>>                    test_database = None,
            >>>                    grid_shape=(30,30,30),
            >>>                    select_feature = {
            >>>                       'AtomicDensities': 'all',
            >>>                       'Features': 'all'
            >>>                    },
            >>>                    select_target='class',
            >>>                    normalize_features = True,
            >>>                    pair_chain_feature=None,
            >>>                    dict_filter={'class': '==0.0'},
            >>>                    process = True)
        '''

        # allow for multiple database
        self.train_database = self._get_database_name(train_database)

        # allow for multiple database
        self.valid_database = self._get_database_name(valid_database)

        # allow for multiple database
        self.test_database = self._get_database_name(test_database)

        # pdb selection
        self.use_rotation = use_rotation

        # features/targets selection
        self.select_feature = select_feature
        self.select_target = select_target

        # map generation
        self.grid_info = grid_info

        # data agumentation
        self.data_augmentation = 0

        # normalization conditions
        self.normalize_features = normalize_features

        # clip the data
        self.clip_features = clip_features
        self.clip_factor = clip_factor

        # shape of the data
        self.input_shape = None
        self.data_shape = None
        self.grid_shape = grid_shape

        # get the eventual projection
        self.transform = transform_to_2D
        self.proj2D = projection

        # filter the dataset
        self.dict_filter = dict_filter

        # print the progress bar or not
        self.tqdm = tqdm

        # set random seed
        self.rotation_seed = rotation_seed

        # process the data
        if process:
            self.process_dataset()

    @staticmethod
    def _get_database_name(database):
        """Get the list of hdf5 database file names.

        Args:
            database(None, str or list(str)): hdf5 database name(s).

        Returns:
            list: hdf5 file names
        """
        # make sure the return is only one data type
        filenames = []
        if database is not None:
            if not isinstance(database, list):
                database = [database]
            for db in database:
                filenames += glob.glob(db)

        return filenames

    def process_dataset(self):
        """Process the data set.

        Done by default. However must be turned off when one want to
        test a pretrained model. This can be done by setting
        ``process=False`` in the creation of the ``DataSet`` instance.
        """

        logger.info('\n')
        logger.info('=' * 40)
        logger.info('=\t DeepRank Data Set')
        logger.info('=')
        logger.info('=\t Training data')
        for f in self.train_database:
            logger.info(f'=\t -> {f}')
        logger.info('=')
        if self.valid_database:
            logger.info('=\t Validation data')
            for f in self.valid_database:
                logger.info(f'=\t -> {f}')
        logger.info('=')
        if self.test_database:
            logger.info('=\t Test data')
            for f in self.test_database:
                logger.info(f'=\t -> {f}')
        logger.info('=')
        logger.info('=' * 40 + '\n')
        sys.stdout.flush()

        # check if the files are ok
        self.check_hdf5_files(self.train_database)

        if self.valid_database:
            self.valid_database = self.check_hdf5_files(
                self.valid_database)

        if self.test_database:
            self.test_database = self.check_hdf5_files(
                self.test_database)

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self.create_index_molecules()

        # get the actual feature name
        self.get_mapped_feature_name()

        # get grid shape
        self.get_grid_shape()

        # get the input shape
        self.get_input_shape()

        # get renormalization factor
        if self.normalize_features:
            self.get_norm()

        logger.info('\n')
        logger.info("   Data Set Info:")
        logger.info(
            f'   Augmentation       : {self.use_rotation} rotations')
        logger.info(
            f'   Training set       : {self.ntrain} conformations')
        logger.info(
            f'   Validation set     : {self.nvalid} conformations')
        logger.info(
            f'   Test set           : {self.ntest} conformations')
        logger.info(f'   Number of channels : {self.input_shape[0]}')
        logger.info(f'   Grid Size          : {self.data_shape[1]}, '
                    f'{self.data_shape[2]}, {self.data_shape[3]}')
        sys.stdout.flush()

    def __len__(self):
        """Get the length of the dataset
        Returns:
            int: number of complexes in the dataset
        """
        return len(self.index_complexes)

    def __getitem__(self, index):
        """Get one item from its unique index.

        Args:
            index (int): index of the complex

        Returns:
            dict: {'mol':[fname,mol],'feature':feature,'target':target}
        """
        fname, mol, angle, axis = self.index_complexes[index]
        try:

            feature, target = self.load_one_variant(fname, mol)

            if self.clip_features:
                feature = self._clip_feature(feature)

            if self.normalize_features:
                feature = self._normalize_feature(feature)

            if self.transform:
                feature = self.convert2d(feature, self.proj2D)

            result = {'mol': [fname, mol], 'feature': feature, 'target': target}
            return result

        except:
            logger.error('Unable to load molecule %s from %s' % (mol, fname))
            raise

    @staticmethod
    def check_hdf5_files(database):
        """Check if the data contained in the hdf5 file is ok."""

        logger.info("   Checking dataset Integrity")
        remove_file = []
        for fname in database:
            try:
                f = h5py.File(fname, 'r')
                mol_names = list(f.keys())
                if len(mol_names) == 0:
                    warnings.warn('    -> %s is empty ' % fname)
                    remove_file.append(fname)
                f.close()
            except BaseException:
                warnings.warn('    -> %s is corrputed ' % fname)
                remove_file.append(fname)

        for name in remove_file:
            database.remove(name)
        if remove_file:
            logger.info(f'\t -> Empty or corrput databases are removed:\n'
                        f'{remove_file}')

        return database

    def create_index_molecules(self):
        """Create the indexing of each molecule in the dataset.

        Create the indexing:
        [('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one complex with its index in the list.

        Raises:
            ValueError: No aviable training data after filtering.
        """
        logger.info("\n\n   Processing data set:")

        self.index_complexes = []

        # Training dataset
        desc = '{:25s}'.format('   Train dataset')
        if self.tqdm:
            data_tqdm = tqdm(self.train_database,
                             desc=desc, file=sys.stdout)
        else:
            logger.info('   Train dataset')
            data_tqdm = self.train_database
        sys.stdout.flush()

        for fdata in data_tqdm:
            if self.tqdm:
                data_tqdm.set_postfix(mol=os.path.basename(fdata))
            try:
                fh5 = h5py.File(fdata, 'r')
                mol_names = list(fh5.keys())
                mol_names = self._select_pdb(mol_names)
                for k in mol_names:
                    if self.filter(fh5[k]):
                        self.index_complexes += [(fdata,
                                                  k, None, None)]
                        for irot in range(self.data_augmentation):
                            axis, angle = pdb2sql.transform.get_rot_axis_angle(
                                self.rotation_seed)
                            self.index_complexes += [
                                (fdata, k, angle, axis)]
                fh5.close()
            except Exception:
                logger.exception(f'Ignore file: {fdata}')

        self.ntrain = len(self.index_complexes)
        self.index_train = list(range(self.ntrain))

        if self.ntrain == 0:
            raise ValueError(
                'No avaiable training data after filtering')

        # Validation dataset
        if self.valid_database:

            desc = '{:25s}'.format('   Validation dataset')
            if self.tqdm:
                data_tqdm = tqdm(self.valid_database,
                                 desc=desc, file=sys.stdout)
            else:
                data_tqdm = self.valid_database
                logger.info('   Validation dataset')
            sys.stdout.flush()

            for fdata in data_tqdm:
                if self.tqdm:
                    data_tqdm.set_postfix(mol=os.path.basename(fdata))
                try:
                    fh5 = h5py.File(fdata, 'r')
                    mol_names = list(fh5.keys())
                    mol_names = self._select_pdb(mol_names)
                    self.index_complexes += [(fdata, k, None, None)
                                             for k in mol_names]
                    fh5.close()
                except Exception:
                    logger.exception(f'Ignore file: {fdata}')

        self.ntot = len(self.index_complexes)
        self.index_valid = list(range(self.ntrain, self.ntot))
        self.nvalid = self.ntot - self.ntrain

        # Test dataset
        if self.test_database:

            desc = '{:25s}'.format('   Test dataset')
            if self.tqdm:
                data_tqdm = tqdm(self.test_database,
                                 desc=desc, file=sys.stdout)
            else:
                data_tqdm = self.test_database
                logger.info('   Test dataset')
            sys.stdout.flush()

            for fdata in data_tqdm:
                if self.tqdm:
                    data_tqdm.set_postfix(mol=os.path.basename(fdata))
                try:
                    fh5 = h5py.File(fdata, 'r')
                    mol_names = list(fh5.keys())
                    mol_names = self._select_pdb(mol_names)
                    self.index_complexes += [(fdata, k, None, None)
                                             for k in mol_names]
                    fh5.close()
                except Exception:
                    logger.exception(f'Ignore file: {fdata}')

        self.ntot = len(self.index_complexes)
        self.index_test = list(
            range(self.ntrain + self.nvalid, self.ntot))
        self.ntest = self.ntot - self.ntrain - self.nvalid

    def _select_pdb(self, mol_names):
        """Select complexes.

        Args:
            mol_names (list): list of complex names

        Returns:
            list: list of selected complexes
        """

        fnames_original = list(
            filter(lambda x: not re.search(r'_r\d+$', x), mol_names))
        if self.use_rotation is not None:
            fnames_augmented = []
            # TODO if there is no augmentation data in dataaset,
            # the fnames_augmented should be 0, should report it.
            if self.use_rotation > 0:
                for i in range(self.use_rotation):
                    fnames_augmented += list(filter(lambda x:
                                                    re.search('_r%03d$' % (i + 1), x), mol_names))
                selected_mol_names = fnames_original + fnames_augmented
            else:
                selected_mol_names = fnames_original
        else:
            selected_mol_names = mol_names
            sample_id = fnames_original[0]
            num_rotations = len(list((filter(lambda x:
                                re.search(sample_id + '_r', x), mol_names))))
            self.use_rotation = num_rotations

        return selected_mol_names

    @staticmethod
    def _insert_before_operators(subject_string, inserting_string):
        """ This function is to insert a string into another string, wherever a '>=', '<=', '!=', '==', '>', '<' operator is encountered.

            Args:
                subject_string (str): the string to insert into
                inserting_string (str): the string to insert

            Returns (str): the string with insertions
        """

        return_string = subject_string
        for operator_string in ['>=', '<=', '==', '!=', '>', '<']:
            search_index = 0
            while search_index < len(return_string):
                found_index = return_string.find(operator_string, search_index)

                if found_index >= 0:
                    # ATTENTION: we don't want to insert it twice!
                    if return_string[:found_index].endswith(inserting_string):

                        search_index = found_index + len(operator_string)
                    else:
                        return_string = return_string[:found_index] + inserting_string + return_string[found_index:]

                        search_index = found_index + len(inserting_string) + len(operator_string)
                else:
                    # not found
                    break

        return return_string

    def filter(self, molgrp):
        """Filter the molecule according to a dictionary, e.g.,
        dict_filter={'class':'==1.0'}

        The filter is based on the attribute self.dict_filter
        that must be either of the form: { 'name': cond } or None

        Args:
            molgrp (hdf5 group): group of the molecule in the hdf5 file
        Returns:
            bool: True if we keep the complex False otherwise

        Raises:
            ValueError: If an unsuported condition is provided
        """

        logger.debug("filtering {}".format(molgrp.name))

        if self.dict_filter is None:
            return True

        for cond_name, cond_vals in self.dict_filter.items():

            logger.debug("filter with condition {}: {}".format(cond_name, cond_vals))

            if cond_name not in molgrp['targets']:
                raise ValueError(f'Filter {cond_name} not found for mol {molgrp.name}')

            val = molgrp['targets/' + cond_name][()]

            # if we have a string it's more complicated
            if isinstance(cond_vals, str):
                new_cond_vals = DataSet._insert_before_operators(cond_vals, 'val')
                if not eval(new_cond_vals):
                    return False
            else:
                raise ValueError("Conditions not supported", cond_vals)

        return True

    def get_mapped_feature_name(self):
        """Get actual mapped feature names for feature selections.

        Note:
            - class parameter self.select_feature examples:
                - 'all'
                - {'AtomicDensities_ind': 'all', 'Feature_ind':all}
                - {'Feature_ind': ['PSSM_*', 'pssm_ic_*']}
            - Feature type must be: 'AtomicDensities_ind' or 'Feature_ind'.

        Raises:
            KeyError: Wrong feature type.
            KeyError: Wrong feature type.
        """

        # open a h5 file in case we need it
        f5 = h5py.File(self.train_database[0], 'r')
        mol_name = list(f5.keys())[0]
        mapped_data = f5.get(mol_name + '/mapped_features/')

        # if we select all the features
        if self.select_feature == "all":

            # redefine dict
            self.select_feature = {}

            # loop over the feat types and add all the feat_names
            for feat_type, feat_names in mapped_data.items():
                self.select_feature[feat_type] = [
                    name for name in feat_names]

        # if a selection was made
        else:

            # we loop over the input dict
            for feat_type, feat_names in self.select_feature.items():

                # if for a given type we need all the feature
                if feat_names == 'all':
                    if feat_type in mapped_data:
                        self.select_feature[feat_type] = list(
                            mapped_data[feat_type].keys())
                    else:
                        self.print_possible_features()
                        raise KeyError('Feature type %s not found' % feat_type)

                else:
                    # TODO to refactor this part
                    if feat_type not in mapped_data:
                        self.print_possible_features()
                        raise KeyError('Feature type %s not found' % feat_type)

                    self.select_feature[feat_type] = []

                    # loop over all the specified feature names
                    for name in feat_names:

                        # if we have a wild card e.g. PSSM_*
                        # we check the matches and add them
                        if '*' in name:
                            match = name.split('*')[0]
                            possible_names = list(
                                mapped_data[feat_type].keys())
                            match_names = [
                                n for n in possible_names
                                if n.startswith(match)]
                            self.select_feature[feat_type] += match_names

                        # if we don't have a wild card we append
                        # <feature_name> to the list
                        else:
                            self.select_feature[feat_type] += [name]

        f5.close()

    def print_possible_features(self):
        """Print the possible features in the group."""

        f5 = h5py.File(self.train_database[0], 'r')
        mol_name = list(f5.keys())[0]
        mapgrp = f5.get(mol_name + '/mapped_features/')

        logger.info('\nPossible Features:')
        logger.info('-' * 20)
        for feat_type in list(mapgrp.keys()):
            logger.info('== %s' % feat_type)
            for fname in list(mapgrp[feat_type].keys()):
                logger.info('   -- %s' % fname)

        if self.select_feature is not None:
            logger.info('\nYour selection was:')
            for feat_type, feat in self.select_feature.items():
                if feat_type not in list(mapgrp.keys()):
                    logger.info(
                        '== \x1b[0;37;41m' + feat_type + '\x1b[0m')
                else:
                    logger.info('== %s' % feat_type)
                    if isinstance(feat, str):
                        logger.info('   -- %s' % feat)
                    if isinstance(feat, list):
                        for f in feat:
                            logger.info('  -- %s' % f)

    def get_input_shape(self):
        """Get the size of the data and input.

        Note:
            - self.data_shape: shape of the raw 3d data set
            - self.input_shape: input size of the CNN. Potentially after 2d transformation.
        """

        fname = self.train_database[0]
        feature, _ = self.load_one_variant(fname)

        self.data_shape = feature.shape

        if self.transform:
            feature = self.convert2d(feature, self.proj2D)

        self.input_shape = feature.shape

    def get_grid_shape(self):
        """Get the shape of the matrices.

        Raises:
            ValueError: If no grid shape is provided or is present in
                the HDF5 file
        """

        fname = self.train_database[0]
        fh5 = h5py.File(fname, 'r')
        mol = list(fh5.keys())[0]

        # get the mol
        mol_data = fh5.get(mol)

        # get the grid size
        if self.grid_shape is None:

            if 'grid_points' in mol_data:
                nx = mol_data['grid_points']['x'].shape[0]
                ny = mol_data['grid_points']['y'].shape[0]
                nz = mol_data['grid_points']['z'].shape[0]
                self.grid_shape = (nx, ny, nz)

            else:
                raise ValueError(
                    f'Impossible to determine sparse grid shape.\n '
                    f'Specify argument grid_shape=(x,y,z)')

            fh5.close()

    def get_norm(self):
        """Get the normalization values for the features."""

        # logger.info("   Normalization factor:")

        # declare the dict of class instance
        # where we'll store the normalization parameter
        self.param_norm = {'features': {}, 'targets': {}}
        for feat_type, feat_names in self.select_feature.items():
            self.param_norm['features'][feat_type] = {}
            for name in feat_names:
                self.param_norm['features'][feat_type][name] = NormParam(
                )
        self.param_norm['targets'][self.select_target] = MinMaxParam()

        # read the normalization
        self._read_norm()

        # make array for fast access
        self.feature_mean, self.feature_std = [], []
        for feat_type, feat_names in self.select_feature.items():
            for name in feat_names:
                self.feature_mean.append(
                    self.param_norm['features'][feat_type][name].mean)
                self.feature_std.append(
                    self.param_norm['features'][feat_type][name].std)

        self.target_min = self.param_norm['targets'][self.select_target].min
        self.target_max = self.param_norm['targets'][self.select_target].max

    def _read_norm(self):
        """Read or create the normalization file for the complex."""
        # loop through all the filename
        for f5 in self.train_database:

            # get the precalculated data
            fdata = os.path.splitext(f5)[0] + '_norm.pckl'

            # if the file doesn't exist we create it
            if not os.path.isfile(fdata):
                logger.info(f"      Computing norm for {f5}")
                norm = NormalizeData(f5, shape=self.grid_shape)
                norm.get()

            # read the data
            data = pickle.load(open(fdata, 'rb'))

            # handle the features
            for feat_type, feat_names in self.select_feature.items():
                for name in feat_names:
                    mean = data['features'][feat_type][name].mean
                    var = data['features'][feat_type][name].var
                    if var == 0:
                        logger.info(
                            ' : STD is null for %s in %s' % (name, f5))
                    self.param_norm['features'][feat_type][name].add(
                        mean, var)

            # handle the target
            minv = data['targets'][self.select_target].min
            maxv = data['targets'][self.select_target].max
            self.param_norm['targets'][self.select_target].update(
                minv)
            self.param_norm['targets'][self.select_target].update(
                maxv)

        # process the std
        nfile = len(self.train_database)
        for feat_types, feat_dict in self.param_norm['features'].items():
            for feat in feat_dict:
                self.param_norm['features'][feat_types][feat].process(
                    nfile)
                if self.param_norm['features'][feat_types][feat].std == 0:
                    logger.info(
                        '  Final STD Null for %s/%s. Changed it to 1' %
                        (feat_types, feat))
                    self.param_norm['features'][feat_types][feat].std = 1

    def backtransform_target(self, data):
        """Returns the values of the target after de-normalization.

        Args:
            data (list(float)): normalized data
        Returns:
            list(float): un-normalized data
        """
        # print(data)
        # print(self.target_max)
        #data = FloatTensor(data)
        data *= self.target_max
        data += self.target_min
        return data  # .numpy()

    def _normalize_feature(self, feature):
        """Normalize the values of the features.

        Args:
            feature (np.array): raw feature values
        Returns:
            np.array: normalized feature values
        """

        for ic in range(self.data_shape[0]):
            feature[ic] = (feature[ic] - self.feature_mean[ic]) / self.feature_std[ic]
        return feature

    def _clip_feature(self, feature):
        """Clip the value of the features at +/- mean + clip_factor * std.
        Args:
            feature (np.array): raw feature values
        Returns:
            np.array: clipped feature values
        """

        w = self.clip_factor
        for ic in range(self.data_shape[0]):
            minv = self.feature_mean[ic] - w * self.feature_std[ic]
            maxv = self.feature_mean[ic] + w * self.feature_std[ic]
            feature[ic] = np.clip(feature[ic], minv, maxv)
            #feature[ic] = self._mad_based_outliers(feature[ic],minv,maxv)
        return feature

    @staticmethod
    def _mad_based_outliers(points, minv, maxv, thresh=3.5):
        """Mean absolute deviation based outlier detection.

        (Experimental).
        Args:
            points (np.array): raw input data
            minv (float): Minimum (negative) value requested
            maxv (float): Maximum (positive) value requested
            thresh (float, optional): Threshold for data detection
        Returns:
            TYPE: data where outliers were replaced by min/max values
        """

        median = np.median(points)
        diff = np.sqrt((points - median)**2)
        med_abs_deviation = np.median(diff)

        if med_abs_deviation == 0:
            return points

        modified_z_score = 0.6745 * diff / med_abs_deviation
        mask_outliers = modified_z_score > thresh

        mask_max = np.abs(points - maxv) < np.abs(points - minv)
        mask_min = np.abs(points - maxv) > np.abs(points - minv)

        points[mask_max * mask_outliers] = maxv
        points[mask_min * mask_outliers] = minv

        return points

    def load_one_variant(self, fname, variant_name=None):
        """Load the feature/target of a single variant.

        Args:
            fname (str): hdf5 file name
            variant_name (None or str, optional): name of the variant in the hdf5

        Returns:
            np.array,float: features, targets
        """
        outtype = 'float32'
        fh5 = h5py.File(fname, 'r')

        if variant_name is None:
            variant_name = list(fh5.keys())[0]

        logger.info("loading variant {} from {}".format(variant_name, fname))

        # get the variant
        variant_data = fh5.get(variant_name)

        # check for mapped features:
        if 'mapped_features' not in variant_data.keys():
            logger.error(f"Error: variant: {variant_name} in {fname} does not have mapped_features ")
            fh5.close()
            sys.exit()

        # get the features
        feature = []
        for feat_type, feat_names in self.select_feature.items():

            logger.debug("selected feature: {} {}".format(feat_type, feat_names))

            # see if the feature exists
            if 'mapped_features/' + feat_type in variant_data.keys():
                feat_dict = variant_data.get('mapped_features/' + feat_type)
            else:
                logger.error(
                    f'Feature type {feat_type} not found in file {fname} '
                    f'for variant {variant_name}.\n'
                    f'Possible feature types are:\n\t' +
                    '\n\t'.join(
                        list(variant_data['mapped_features'].keys()))
                )
                raise ValueError(feat_type, ' not supported')

            # loop through all the desired feat names
            for name in feat_names:

                # extract the group
                try:
                    data = feat_dict[name]
                except KeyError:
                    logger.error(
                        f'Feature {name} not found in file {fname} for variant '
                        f'{variant_name} and feature type {feat_type}.\n'
                        f'Possible feature are:\n\t' +
                        '\n\t'.join(list(
                            variant_data['mapped_features/' +
                                     feat_type].keys()
                        ))
                    )
                    continue

                # check its sparse attribute
                # if true get a FLAN
                # if flase direct import
                if data.attrs['sparse']:
                    mat = sparse.FLANgrid(sparse=True,
                                          index=data['index'][:],
                                          value=data['value'][:],
                                          shape=self.grid_shape).to_dense()
                else:
                    mat = data['value'][:]

                # append to the list of features
                feature.append(mat)

                logger.debug("converted feature {} {} to a {} matrix".format(feat_type, name, "x".join([str(n) for n in mat.shape])))

        # get the target value
        target = variant_data.get('targets/' + self.select_target)[()]

        logger.debug("{} has target {}".format(variant_name, target))

        # close
        fh5.close()

        # make sure all the feature have exact same type
        # if they don't collate_fn in the creation of the minibatch will fail.
        # Note returning torch.FloatTensor makes each epoch twice longer ...
        return (np.array(feature).astype(outtype),
                np.array([target]).astype(outtype))

    @staticmethod
    def convert2d(feature, proj2d):
        """Convert the 3D volumetric feature to a 2D planar data set.

        proj2d specifies the dimension that we want to consider as channel
        for example for proj2d = 0 the 2D images are in the yz plane and
        the stack along the x dimension is considered as extra channels
        Args:
            feature (np.array): raw features
            proj2d (int): projection
        Returns:
            np.array: projected features
        """
        nc, nx, ny, nz = feature.shape
        if proj2d == 0:
            feature = feature.reshape(-1, 1, ny, nz).squeeze()
        elif proj2d == 1:
            feature = feature.reshape(-1, nx, 1, nz).squeeze()
        elif proj2d == 2:
            feature = feature.reshape(-1, nx, ny, 1).squeeze()

        return feature

    @staticmethod
    def _densgrid(center, vdw_radius, grid, npts):
        """Function to map individual atomic density on the grid.

        The formula is equation (1) of the Koes paper
        Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1

        Args:
            center (list(float)): position of the atoms
            vdw_radius (float): vdw radius of the atom

        Returns:
            TYPE: np.array (mapped density)
        """

        x0, y0, z0 = center
        dd = np.sqrt((grid[0] - x0)**2 +
                     (grid[1] - y0)**2 + (grid[2] - z0)**2)
        dgrid = np.zeros(npts)

        dgrid[dd < vdw_radius] = np.exp(
            -2 * dd[dd < vdw_radius]**2 / vdw_radius**2)

        dd_tmp = dd[(dd >= vdw_radius) & (dd < 1.5 * vdw_radius)]
        dgrid[(dd >= vdw_radius) & (dd < 1.5 * vdw_radius)] = (
            4. / np.e**2 / vdw_radius**2 * dd_tmp**2) - (
                12. / np.e**2 / vdw_radius * dd_tmp) + 9. / np.e**2

        return dgrid

    @staticmethod
    def _featgrid(center, value, grid, npts):
        """Map an individual feature (atomic or residue) on the grid.

        Args:
            center (list(float)): position of the feature center
            value (float): value of the feature
            type_ (str, optional): method to map

        Returns:
            np.array: Mapped feature

        Raises:
            ValueError: Description
        """

        # shortcut for th center
        x0, y0, z0 = center

        sigma = np.sqrt(1. / 2)
        beta = 0.5 / (sigma**2)
        cutoff = 5. * beta

        dd = np.sqrt((grid[0] - x0)**2 +
                     (grid[1] - y0)**2 + (grid[2] - z0)**2)

        dd[dd < cutoff] = value * np.exp(-beta * dd[dd < cutoff])
        dd[dd > cutoff] = 0

        #dgrid = np.zeros(npts)
        #dgrid[dd<cutoff] = value*np.exp(-beta*dd[dd<cutoff])
        # print(np.allclose(dgrid,dd))

        return dd
