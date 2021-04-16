import importlib
import copy
import os
import re
import sys
import warnings
from collections import OrderedDict

import h5py
import numpy as np

import deeprank
from deeprank import config
from deeprank.config import logger
from deeprank.generate import GridTools as gt
import pdb2sql
from pdb2sql.align import align as align_along_axis
from pdb2sql.align import align_interface

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

try:
    from pycuda import driver, compiler, gpuarray, tools
    import pycuda.autoinit
except ImportError:
    pass


def _printif(string, cond): return print(string) if cond else None


class DataGenerator(object):

    def __init__(self, mutants,
                 compute_targets=None, compute_features=None,
                 data_augmentation=None, hdf5='database.h5',
                 mpi_comm=None):
        """Generate the data (features/targets/maps) required for deeprank.

        Args:
            mutants (list(PdbMutantSelection)): the selected mutants
            pdb_path (str): file path where to find the pdb for mapping
            pssm_path (str, optional): path where to find the PSSM file
            compute_targets (list(str), optional): List of python files computing the targets,
                "pdb_native" must be set if having targets to compute.
            compute_features (list(str), optional): List of python files computing the features
            data_augmentation (int, optional): Number of rotations performed one each pdb
            hdf5 (str, optional): name of the hdf5 file where the data is saved, default to 'database.h5'
            mpi_comm (MPI_COMM): MPI COMMUNICATOR

        Raises:
            NotADirectoryError: if the source are not found

        Example:

            >>> from deeprank.generate import *
            >>> from deeprank.models.mutant import PdbMutantSelection
            >>> # sources to assemble the data base
            >>> pdb_path = "1AK4.pdb"
            >>> pssm_path = "1AK4.pssm"
            >>> mutant = PdbMutantSelection("C", 450, "A")
            >>> h5file = '1ak4.hdf5'
            >>>
            >>> #init the data assembler
            >>> database = DataGenerator(mutant, pdb_path, pssm_path,
            >>>                          compute_targets=['deeprank.targets.dockQ'],
            >>>                          compute_features=['deeprank.features.AtomicFeature',
            >>>                                            'deeprank.features.PSSM_IC',
            >>>                                            'deeprank.features.BSA'],
            >>>                          hdf5=h5file)
        """

        self.mutants = mutants

        self.compute_targets = compute_targets
        self.compute_features = compute_features

        self.data_augmentation = data_augmentation

        self.hdf5 = hdf5

        self.mpi_comm = mpi_comm

        self.feature_error = []
        self.grid_error = []
        self.map_error = []

        self.logger = logger

        # handle pssm source
        pssm_features = ('deeprank.features.FullPSSM',
                         'deeprank.features.PSSM_IC')
        if self.compute_features and \
                set.intersection(set(pssm_features), set(self.compute_features)):
            if config.PATH_PSSM_SOURCE is None:
                raise ValueError(
                    'You must provide "pssm_source" to compute PSSM features.')

# ====================================================================================
#
#       CREATE THE DATABASE ALL AT ONCE IF ALL OPTIONS ARE GIVEN
#
# ====================================================================================

    def create_database(
            self,
            verbose=False,
            remove_error=True,
            prog_bar=False,
            contact_distance=8.5,
            random_seed=None):
        """Create the hdf5 file architecture and compute the features/targets.

        Args:
            verbose (bool, optional): Print creation details
            remove_error (bool, optional): remove the groups that errored
            prog_bar (bool, optional): use tqdm
            contact_distance (float): contact distance cutoff, defaults to 8.5Å
            random_seed (int): random seed for getting rotation axis and angle

        Raises:
            ValueError: If creation of the group errored.

        Example:

        >>> # sources to assemble the data base
        >>> pdb_path = "1AK4.pdb"
        >>> h5file = '1ak4.hdf5'
        >>> mutant = PdbMutantSelection("C", 450, "A")
        >>>
        >>> #init the data assembler
        >>> database = DataGenerator(mutant, pdb_path, pssm_path,
        >>>                          data_augmentation=None,
        >>>                          compute_targets  = ['deeprank.targets.dockQ'],
        >>>                          compute_features = ['deeprank.features.AtomicFeature',
        >>>                                              'deeprank.features.PSSM_IC',
        >>>                                              'deeprank.features.BSA'],
        >>>                          hdf5=h5file)
        >>>
        >>> #create new files
        >>> database.create_database(prog_bar=True)
        """
        # check decoy pdb files
        if not self.pdb_path:
            raise ValueError(f"no pdb file. Check class parameter 'pdb_path'")

        self.local_mutants = self.mutants

        if self.mpi_comm is not None:
            rank = self.mpi_comm.Get_rank()
            size = self.mpi_comm.Get_size()
        else:
            size = 1

        if size > 1:
            if rank == 0:
                mutants_divided = [self.mutants[i::size] for i in range(size)]
                self.local_mutants = mutant_divided[0]
                # send to other procs
                for iP in range(1, size):
                    self.mpi_comm.send(mutants_divided[iP], dest=iP, tag=11)
            else:
                # receive procs
                self.local_mutants = self.mpi_comm.recv(source=0, tag=11)
            # change hdf5 name
            h5path, h5name = os.path.split(self.hdf5)
            self.hdf5 = os.path.join(h5path, f"{rank:03d}_{h5name}")

        # open the file
        self.f5 = h5py.File(self.hdf5, 'w')

        # set metadata to hdf5 file
        self.f5.attrs['DeepRank_version'] = deeprank.__version__
        if self.compute_features is not None:
            self.f5.attrs['features'] = self.compute_features
        if self.compute_targets is not None:
            self.f5.attrs['targets'] = self.compute_targets

        ##################################################
        # Start generating HDF5 database
        ##################################################
        self.logger.info(
            f'\n# Start creating HDF5 database: {self.hdf5}')

        # get the local progress bar
        desc = '{:25s}'.format('Creating database')
        mutant_tqdm = tqdm(self.local_mutants,
                         desc=desc,
                         disable=not prog_bar)

        for mutant in mutant_tqdm:

            mutant_tqdm.set_postfix(mol=os.path.basename(mutant.pdb_path))
            self.logger.info('\nProcessing PDB file: {}'.format(mutant.pdb_path))

            # names of the molecule
            mol_name = os.path.splitext(os.path.basename(mutant.pdb_path))[0]
            mol_aug_name_list = []

            try:

                ################################################
                #   get the pdbs of the conformation and its ref
                #   for the original data (not augmetned one)
                ################################################

                if verbose:
                    self.logger.info(
                        f'\nMolecule: {mol_name}.'
                        f'\nStart generating top HDF5 group "{mol_name}"...'
                        f'\n{"":4s}Reading PDB data into database...')

                # create a subgroup for the molecule
                molgrp = self.f5.require_group(mol_name)
                molgrp.attrs['type'] = 'molecule'

                self._add_pdb(molgrp, mutant.pdb_path, "pdb")
                if mutant.pssm_path is not None:
                    self._add_pssm(molgrp, mutant.pssm_path, "pssm")

                if verbose:
                    self.logger.info(
                        f'{"":4s}Generated subgroup "pdb"'
                        f' to store pdb data of the current model.')

                ################################################
                #   add the features
                ################################################
                feature_error_flag = False  # when False: success; when True: failed

                if self.compute_features is not None:
                    if verbose:
                        self.logger.info(
                            f'{"":4s}Calculating features...')

                    molgrp.require_group('features')
                    molgrp.require_group('features_raw')

                    feature_error_flag = self._compute_features(self.compute_features,
                                                                molgrp['pdb'][(
                                                                )],
                                                                molgrp['features'],
                                                                molgrp['features_raw'],
                                                                self.mutant,
                                                                self.logger)
                    if feature_error_flag:
                        self.feature_error += [mol_name]
                        # ignore the targets/grid/augmentation computation
                        # and directly go to next molecule. Remove errored
                        # molecule later.
                        # Otherwise, keep computing and report errored mol.
                        if remove_error:
                            continue

                    if verbose:
                        if not feature_error_flag or not remove_error:
                            self.logger.info(
                                f'\n{"":4s}Generated subgroup "features"'
                                f' to store xyz-based feature values.'
                                f'{"":4s}Generated subgroup "features_raw"'
                                f' to store human read feature values')

                ################################################
                #   add the targets
                ################################################
                if self.compute_targets is not None:
                    if verbose:
                        self.logger.info(
                            f'{"":4s}Calculating targets...')

                    molgrp.require_group('targets')

                    self._compute_targets(self.compute_targets,
                                          molgrp['pdb'][()],
                                          molgrp['targets'])

                    if verbose:
                        self.logger.info(
                            f'{"":4s}Generated subgroup "targets" '
                            f'to store targets, such as BIN_CLASS, dockQ, etc.')

                ################################################
                #   add the box center
                ################################################
                if verbose:
                    self.logger.info(
                        f'{"":4s}Calculating grid box center...')

                grid_error_flag = False
                molgrp.require_group('grid_points')

                try:
                    center = self._get_grid_center(
                        molgrp['pdb'][()], contact_distance)
                    molgrp['grid_points'].create_dataset(
                        'center', data=center)
                    if verbose:
                        self.logger.info(
                            f'{"":4s}Generated subgroup "grid_points"'
                            f' to store grid box center.')
                except ValueError as ex:
                    grid_error_flag = True
                    self.grid_error += [mol_name]
                    self.logger.exception(ex)
                    if remove_error:
                        continue

                ################################################
                #   DATA AUGMENTATION
                ################################################

                # GET ALL THE NAMES
                if self.data_augmentation is not None:
                    mol_aug_name_list = [
                        mol_name +
                        '_r%03d' %
                        (idir +
                         1) for idir in range(
                            self.data_augmentation)]
                else:
                    mol_aug_name_list = []

                if verbose and mol_aug_name_list:
                    self.logger.info(
                        f'{"":2s}Start augmenting data'
                        f' with {self.data_augmentation} times...')

                # loop over the complexes
                for mol_aug_name in mol_aug_name_list:

                    # create a subgroup for the molecule
                    molgrp = self.f5.require_group(mol_aug_name)
                    molgrp.attrs['type'] = 'molecule'

                    # get the rotation axis and angle
                    if self.align is None:
                        axis, angle = pdb2sql.transform.get_rot_axis_angle(random_seed)
                    else:
                        axis, angle = self._get_aligned_rotation_axis_angle(random_seed, self.align)

                    # create the new pdb and get molecule center
                    # molecule center is the origin of rotation)
                    mol_center = self._add_aug_pdb(molgrp, mutant.pdb_path, 'pdb', axis, angle)

                    # copy the targets/features
                    if 'targets' in self.f5[mol_name]:
                        self.f5.copy(mol_name + '/targets/', molgrp)
                    self.f5.copy(mol_name + '/features/', molgrp)

                    # rotate the feature
                    self._rotate_feature(
                        molgrp, axis, angle, mol_center)

                    # grid center used to create grid box
                    molgrp.require_group('grid_points')
                    center = pdb2sql.transform.rot_xyz_around_axis(self.f5[mol_name + '/grid_points/center'],
                                                                   axis, angle, mol_center)

                    molgrp['grid_points'].create_dataset('center', data=center)

                    # store the rotation axis/angl/center as attriutes
                    # in case we need them later
                    molgrp.attrs['axis'] = axis
                    molgrp.attrs['angle'] = angle
                    molgrp.attrs['center'] = mol_center

                # cache aug mols if original mol has errored features
                if feature_error_flag:
                    self.feature_error += mol_aug_name_list
                if grid_error_flag:
                    self.grid_error += mol_aug_name_list

                if verbose and mol_aug_name_list:
                    self.logger.info(
                        f'{"":2s}Completed data augmentation'
                        f' and generated top HDF5 groups, e.g. {mol_aug_name}.')

                ################################################
                # Successul message
                ################################################
                if verbose:
                    self.logger.info(
                        f'\nSuccessfully generated top HDF5 group "{mol_name}".\n')

            # all other errors
            except BaseException:
                raise

        ##################################################
        # Post processing
        ##################################################
        #  Remove errored molecules
        errored_mol = list(set(self.feature_error + self.grid_error))
        if errored_mol:
            if remove_error:
                for mol in errored_mol:
                    del self.f5[mol]
                if self.feature_error:
                    self.logger.info(
                        f'Molecules with errored features are removed:'
                        f'\n{self.feature_error}')
                if self.grid_error:
                    self.logger.info(
                        f'Molecules with errored grid points are removed:'
                        f'\n{self.grid_error}')
            else:
                if self.feature_error:
                    self.logger.warning(
                        f'The following molecules have errored features:'
                        f'\n{self.feature_error}')
                if self.grid_error:
                    self.logger.warning(
                        f'The following molecules have errored grid points:'
                        f'\n{self.grid_error}')

        # close the file
        self.f5.close()
        self.logger.info(
            f'\n# Successfully created database: {self.hdf5}\n')

    def aug_data(self, augmentation, keep_existing_aug=True, random_seed=None):
        """Augment exiting original PDB data and features.

        Args:
            augmentation(int): Times of augmentation
            keep_existing_aug (bool, optional): Keep existing augmentated data.
                If False, existing aug will be removed. Defaults to True.

        Examples:
            >>> database = DataGenerator(h5='database.h5')
            >>> database.aug_data(augmentation=3, append=True)
            >>> grid_info = {
            >>>     'number_of_points': [30,30,30],
            >>>     'resolution': [1.,1.,1.],
            >>>     'atomic_densities': {'C':1.7, 'N':1.55, 'O':1.52, 'S':1.8},
            >>>     }
            >>> database.map_features(grid_info)
        """

        # check if file exists
        if not os.path.isfile(self.hdf5):
            raise FileNotFoundError(
                'File %s does not exists' % self.hdf5)

        # get the folder names
        f5 = h5py.File(self.hdf5, 'a')
        fnames = f5.keys()

        # get the non rotated ones
        fnames_original = list(
            filter(lambda x: not re.search(r'_r\d+$', x), fnames))

        # get the rotated ones
        fnames_augmented = list(
            filter(lambda x: re.search(r'_r\d+$', x), fnames))

        aug_id_start = 0
        if keep_existing_aug:
            exiting_augs = list(
                filter(lambda x: re.search(fnames_original[0] + r'_r\d+$', x), fnames_augmented))
            aug_id_start += len(exiting_augs)
        else:
            for i in fnames_augmented:
                del f5[i]

        self.logger.info(
            f'{"":s}\n# Start augmenting data'
            f' with {augmentation} times...')

        # GET ALL THE NAMES
        for mol_name in fnames_original:
            mol_aug_name_list = [
                mol_name + '_r%03d' % (idir + 1) for idir in
                range(aug_id_start, aug_id_start + augmentation)]

            # loop over the pdbs
            for mol_aug_name in mol_aug_name_list:

                # crete a subgroup for the molecule
                molgrp = f5.require_group(mol_aug_name)
                molgrp.attrs['type'] = 'molecule'

                # copy the ref into it
                if 'native' in f5[mol_name]:
                    f5.copy(mol_name + '/native', molgrp)

                # get the rotation axis and angle
                if self.align is None:
                    axis, angle = pdb2sql.transform.get_rot_axis_angle(
                        random_seed)
                else:
                    axis, angle = self._get_aligned_rotation_axis_angle(random_seed,
                                                                        self.align)

                # create the new pdb and get molecule center
                # molecule center is the origin of rotation)
                mol_center = self._add_aug_pdb(
                    molgrp, f5[mol_name + '/pdb'][()], 'pdb', axis, angle)

                # copy the targets/features
                if 'targets' in f5[mol_name]:
                    f5.copy(mol_name + '/targets/', molgrp)
                f5.copy(mol_name + '/features/', molgrp)

                # rotate the feature
                self._rotate_feature(molgrp, axis, angle, mol_center)

                # grid center used to create grid box
                molgrp.require_group('grid_points')
                center = pdb2sql.transform.rot_xyz_around_axis(
                    f5[mol_name + '/grid_points/center'],
                    axis, angle, mol_center)

                molgrp['grid_points'].create_dataset(
                    'center', data=center)

                # store the rotation axis/angl/center as attriutes
                # in case we need them later
                molgrp.attrs['axis'] = axis
                molgrp.attrs['angle'] = angle
                molgrp.attrs['center'] = mol_center
        f5.close()
        self.logger.info(
            f'\n# Successfully augmented data in {self.hdf5}')

# ====================================================================================
#
#       ADD FEATURES TO AN EXISTING DATASET
#
# ====================================================================================

    def add_feature(self, remove_error=True, prog_bar=True):
        """Add a feature to an existing hdf5 file.

        Args:
            remove_error (bool): remove errored molecule
            prog_bar (bool, optional): use tqdm

        Example:

        >>> h5file = '1ak4.hdf5'
        >>>
        >>> #init the data assembler
        >>> database = DataGenerator(compute_features  = ['deeprank.features.ResidueDensity'],
        >>>                          hdf5=h5file)
        >>>
        >>> database.add_feature(remove_error=True, prog_bar=True)
        """

        # check if file exists
        if not os.path.isfile(self.hdf5):
            raise FileNotFoundError(
                'File %s does not exists' % self.hdf5)

        # get the folder names
        f5 = h5py.File(self.hdf5, 'a')
        fnames = f5.keys()

        # get the non rotated ones
        fnames_original = list(
            filter(lambda x: not re.search(r'_r\d+$', x), fnames))

        # get the rotated ones
        fnames_augmented = list(
            filter(lambda x: re.search(r'_r\d+$', x), fnames))

        # check feature_error
        if not self.feature_error:
            self.feature_error = []

        # computes the features of the original
        desc = '{:25s}'.format('Add features')
        for cplx_name in tqdm(
                fnames_original,
                desc=desc,
                ncols=100,
                disable=not prog_bar):

            # molgrp
            molgrp = f5[cplx_name]

            error_flag = False
            if self.compute_features is not None:

                # the internal features
                molgrp.require_group('features')
                molgrp.require_group('features_raw')

                error_flag = self._compute_features(self.compute_features,
                                                    molgrp['pdb'][()],
                                                    molgrp['features'],
                                                    molgrp['features_raw'],
                                                    self.mutant,
                                                    self.logger)

                if error_flag:
                    self.feature_error += [cplx_name]

        # copy the data from the original to the augmented
        for cplx_name in fnames_augmented:

            # group of the molecule
            aug_molgrp = f5[cplx_name]

            # get the source group
            mol_name = re.split(r'_r\d+', molgrp.name)[0]
            src_molgrp = f5[mol_name]

            # get the rotation parameters
            axis = aug_molgrp.attrs['axis']
            angle = aug_molgrp.attrs['angle']
            center = aug_molgrp.attrs['center']

            # copy the features to the augmented
            for k in molgrp['features']:
                if k not in aug_molgrp['features']:

                    # copy
                    data = src_molgrp['features/' + k][()]
                    aug_molgrp.require_group('features')
                    aug_molgrp.create_dataset(
                        "features/" + k, data=data)

                    # rotate
                    self._rotate_feature(
                        aug_molgrp, axis, angle, center, feat_name=[k])

        # find errored augmented molecules
        tmp_aug_error = []
        for mol in self.feature_error:
            tmp_aug_error += list(filter(lambda x: mol in x,
                                         fnames_augmented))
        self.feature_error += tmp_aug_error

        #  Remove errored molecules
        if self.feature_error:
            if remove_error:
                for mol in self.feature_error:
                    del f5[mol]
                self.logger.info(
                    f'Molecules with errored features are removed:\n'
                    f'{self.feature_error}')
            else:
                self.logger.warning(
                    f"The following molecules has errored features:\n"
                    f'{self.feature_error}')

        # close the file
        f5.close()

# ====================================================================================
#
#       ADD TARGETS TO AN EXISTING DATASET
#
# ====================================================================================

    def add_unique_target(self, targdict):
        """Add identical targets for all the pdbs in the datafile.

        This is usefull if you want to add the binary class of all the pdbs
        created from decoys or natives

        Args:
            targdict (dict): Example: {'DOCKQ':1.0}

        >>> database = DataGenerator(hdf5='1ak4.hdf5')
        >>> database.add_unique_target({'DOCKQ':1.0})
        """

        # check if file exists
        if not os.path.isfile(self.hdf5):
            raise FileNotFoundError(
                'File %s does not exists' % self.hdf5)

        f5 = h5py.File(self.hdf5, 'a')
        for mol in list(f5.keys()):
            targrp = f5[mol].require_group('targets')
            for name, value in targdict.items():
                targrp.create_dataset(name, data=np.array([value]))
        f5.close()

    def add_target(self, prog_bar=False):
        """Add a target to an existing hdf5 file.

        Args:
            prog_bar (bool, optional): Use tqdm

        Example:

        >>> h5file = '1ak4.hdf5'
        >>>
        >>> #init the data assembler
        >>> database = DataGenerator(compute_targets =['deeprank.targets.binary_class'],
        >>>                          hdf5=h5file)
        >>>
        >>> database.add_target(prog_bar=True)
        """

        # check if file exists
        if not os.path.isfile(self.hdf5):
            raise FileNotFoundError(
                'File %s does not exists' % self.hdf5)

        # name of the hdf5 file
        f5 = h5py.File(self.hdf5, 'a')

        # get the folder names
        fnames = f5.keys()

        # get the non rotated ones
        fnames_original = list(
            filter(lambda x: not re.search(r'_r\d+$', x), fnames))
        fnames_augmented = list(
            filter(lambda x: re.search(r'_r\d+$', x), fnames))

        # compute the targets  of the original
        desc = '{:25s}'.format('Add targets')

        for cplx_name in tqdm(fnames_original, desc=desc,
                              ncols=100, disable=not prog_bar):

            # group of the molecule
            molgrp = f5[cplx_name]

            # add the targets
            if self.compute_targets is not None:

                molgrp.require_group('targets')
                self._compute_targets(self.compute_targets,
                                      molgrp['pdb'][()],
                                      molgrp['targets'])

        # copy the targets of the original to the rotated
        for cplx_name in fnames_augmented:

            # group of the molecule
            aug_molgrp = f5[cplx_name]

            # get the source group
            mol_name = re.split(r'_r\d+', molgrp.name)[0]
            src_molgrp = f5[mol_name]

            # copy the targets to the augmented
            for k in molgrp['targets']:
                if k not in aug_molgrp['targets']:
                    data = src_molgrp['targets/' + k][()]
                    aug_molgrp.require_group('targets')
                    aug_molgrp.create_dataset(
                        "targets/" + k, data=data)

        # close the file
        f5.close()

    def realign_pdbs(self, align, compute_features=None, pssm_source=None):
        """Align all the pdbs already present in the HDF5.

        Arguments:
            align {dict} -- alignement dictionary (see __init__)

        Keyword Arguments:
            compute_features {list} -- list of features to be computed
                                       if None computes the features specified in
                                       the attrs['features'] of the file (if present)
             pssm_source {str} -- path of the pssm files. If None the source specfied in
                                  the attrs['pssm_source'] will be used (if present) (default: {None})

        Raises:
            ValueError: If no PSSM detected

        Example:

        >>> database = DataGenerator(hdf5='1ak4.hdf5')
        >>> # if comute_features and pssm_source are not specified
        >>> # the values in hdf5.attrs['features'] and hdf5.attrs['pssm_source'] will be used
        >>> database.realign_complex(align={'axis':'x'},
        >>>                          compute_features['deeprank.features.X'],
        >>>                           pssm_source='./1ak4_pssm/')
        """

        f5 = h5py.File(self.hdf5, 'a')

        mol_names = f5.keys()
        self.logger.info(
            f'\n# Start aligning the HDF5 database: {self.hdf5}')

        # deal with the features
        if self.compute_features is None:
            if compute_features is None:
                if 'features' in f5.attrs:
                    self.compute_features = list(f5.attrs['features'])
            else:
                self.compute_features = compute_features

        # deal with the pssm source
        if self.pssm_source is not None:
            config.PATH_PSSM_SOURCE = self.pssm_source

        elif pssm_source is not None:
            config.PATH_PSSM_SOURCE = pssm_source

        elif 'pssm_source' in f5.attrs:
            config.PATH_PSSM_SOURCE = f5.attrs['pssm_source']
        else:
            raise ValueError('No pssm source detected')

        # loop over the pdbs
        desc = '{:25s}'.format('Add features')
        for mol in tqdm(mol_names, desc=desc, ncols=100):

            # align the pdb
            molgrp = f5[mol]
            pdb = molgrp['pdb'][()]

            sqldb = self._get_aligned_sqldb(pdb, align)
            data = sqldb.sql2pdb()

            data = np.array(data).astype('|S78')
            molgrp['pdb'][...] = data

            # remove prexisting features
            old_dir = ['features', 'features_raw', 'mapped_features']
            for od in old_dir:
                if od in molgrp:
                    del molgrp[od]

            # the internal features
            molgrp.require_group('features')
            molgrp.require_group('features_raw')

            # compute features
            error_flag = self._compute_features(self.compute_features,
                                                molgrp['pdb'][()],
                                                molgrp['features'],
                                                molgrp['features_raw'],
                                                self.mutant,
                                                self.logger)

        f5.close()

# ====================================================================================
#
#       PRECOMPUTE THE GRID POINTS
#
# ====================================================================================

    def _get_grid_center(self, pdb, contact_distance):

        sqldb = pdb2sql.interface(pdb)
        # Get the position of the C-alpha
        center_pos = sqldb.get('x,y,z', resSeq=self.mutant.residue_number,
                                        chainID=self.mutant.chain_id,
                                        name="CA")

        sqldb._close()

        return center_pos

    def precompute_grid(self,
                        grid_info,
                        contact_distance=8.5,
                        prog_bar=False,
                        time=False,
                        try_sparse=True):

        # name of the hdf5 file
        f5 = h5py.File(self.hdf5, 'a')

        # check all the input PDB files
        mol_names = f5.keys()

        # get the local progress bar
        desc = '{:25s}'.format('Precompute grid points')
        mol_tqdm = tqdm(mol_names, desc=desc, disable=not prog_bar)

        if not prog_bar:
            print(desc, ':', self.hdf5)
            sys.stdout.flush()

        # loop over the data files
        for mol in mol_tqdm:

            mol_tqdm.set_postfix(mol=mol)

            # compute the data we want on the grid
            gt.GridTools(molgrp=f5[mol],
                         mutant=self.mutant,
                         number_of_points=grid_info['number_of_points'],
                         resolution=grid_info['resolution'],
                         contact_distance=contact_distance,
                         time=time,
                         prog_bar=prog_bar,
                         try_sparse=try_sparse)

        f5.close()


# ====================================================================================
#
#       MAP THE FEATURES TO THE GRID
#
# ====================================================================================


    def map_features(self, grid_info={},
                     cuda=False, gpu_block=None,
                     cuda_kernel='kernel_map.c',
                     cuda_func_name='gaussian',
                     try_sparse=True,
                     reset=False, use_tmpdir=False,
                     time=False,
                     prog_bar=True, grid_prog_bar=False,
                     remove_error=True):
        """Map the feature on a grid of points centered at the interface.

        If features to map are not given, they will be are automatically
        determined for each molecule. Otherwise, given features will be mapped
        for all molecules (i.e. existing mapped features will be recalculated).

        Args:
            grid_info (dict): Informaton for the grid.
                See deeprank.generate.GridTools.py for details.
            cuda (bool, optional): Use CUDA
            gpu_block (None, optional): GPU block size to be used
            cuda_kernel (str, optional): filename containing CUDA kernel
            cuda_func_name (str, optional): The name of the function in the kernel
            try_sparse (bool, optional): Try to save the grids as sparse format
            reset (bool, optional): remove grids if some are already present
            use_tmpdir (bool, optional): use a scratch directory
            time (bool, optional): time the mapping process
            prog_bar (bool, optional): use tqdm for each molecule
            grid_prog_bar (bool, optional): use tqdm for each grid
            remove_error (bool, optional): remove the data that errored

        Example:

        >>> #init the data assembler
        >>> database = DataGenerator(hdf5='1ak4.hdf5')
        >>>
        >>> # map the features
        >>> grid_info = {
        >>>     'number_of_points': [30,30,30],
        >>>     'resolution': [1.,1.,1.],
        >>>     'atomic_densities': {'C':1.7, 'N':1.55, 'O':1.52, 'S':1.8},
        >>> }
        >>>
        >>> database.map_features(grid_info,try_sparse=True,time=False,prog_bar=True)
        """

        # default CUDA
        cuda_func = None
        cuda_atomic = None

        # disable CUDA when using MPI
        if self.mpi_comm is not None:
            if self.mpi_comm.Get_size() > 1:
                if cuda:
                    self.logger.warning(
                        'CUDA mapping disabled when using MPI')
                    cuda = False

        # name of the hdf5 file
        f5 = h5py.File(self.hdf5, 'a')

        # check all the input PDB files
        mol_names = f5.keys()

        if len(mol_names) == 0:
            f5.close()
            raise ValueError(f'No molecules found in {self.hdf5}.')

        ################################################################
        # Check grid_info
        ################################################################
        # fills in the grid data if not provided: default = NONE
        grid_info_ref = copy.deepcopy(grid_info)
        grinfo = ['number_of_points', 'resolution']
        for gr in grinfo:
            if gr not in grid_info:
                grid_info[gr] = None

        # by default we do not map atomic densities
        if 'atomic_densities' not in grid_info:
            grid_info['atomic_densities'] = None

        # fills in the features mode if somes are missing: default = IND
        modes = ['atomic_densities_mode', 'feature_mode']
        for m in modes:
            if m not in grid_info:
                grid_info[m] = 'ind'

        ################################################################
        #
        ################################################################
        # sanity check for cuda
        if cuda and gpu_block is None:  # pragma: no cover
            self.logger.info(
                f'GPU block automatically set to 8 x 8 x 8. '
                f'You can set block size with gpu_block=[n,m,k]')
            gpu_block = [8, 8, 8]

        # initialize cuda
        if cuda:  # pragma: no cover

            # compile cuda module
            npts = grid_info['number_of_points']
            res = grid_info['resolution']
            module = self._compile_cuda_kernel(cuda_kernel, npts, res)

            # get the cuda function for the atomic/residue feature
            cuda_func = self._get_cuda_function(
                module, cuda_func_name)

            # get the cuda function for the atomic densties
            cuda_atomic_name = 'atomic_densities'
            cuda_atomic = self._get_cuda_function(
                module, cuda_atomic_name)

        # get the local progress bar
        desc = '{:25s}'.format('Map Features')
        mol_tqdm = tqdm(mol_names, desc=desc, disable=not prog_bar)

        if not prog_bar:
            self.logger.info(f'{desc}: {self.hdf5}')

        # loop over the data files
        for mol in mol_tqdm:
            mol_tqdm.set_postfix(mol=mol)

            # Determine which feature to map
            # if feature not given, then determine it for each molecule
            if 'feature' not in grid_info_ref:
                # if we havent mapped anything yet or if we reset
                if 'mapped_features' not in list(f5[mol].keys()) or reset:
                    grid_info['feature'] = list(
                        f5[mol + '/features'].keys())

                # if we have already mapped stuff
                elif 'mapped_features' in list(f5[mol].keys()):

                    # feature name
                    all_feat = list(f5[mol + '/features'].keys())

                    # feature already mapped
                    mapped_feat = list(
                        f5[mol + '/mapped_features/Feature_ind'].keys())

                    # we select only the feture that were not mapped yet
                    grid_info['feature'] = []
                    for feat_name in all_feat:
                        if not any(map(lambda x: x.startswith(feat_name + '_'),
                                       mapped_feat)):
                            grid_info['feature'].append(feat_name)

            try:
                # compute the data we want on the grid
                gt.GridTools(
                    molgrp=f5[mol],
                    mutant=self.mutant,
                    number_of_points=grid_info['number_of_points'],
                    resolution=grid_info['resolution'],
                    atomic_densities=grid_info['atomic_densities'],
                    atomic_densities_mode=grid_info['atomic_densities_mode'],
                    feature=grid_info['feature'],
                    feature_mode=grid_info['feature_mode'],
                    cuda=cuda,
                    gpu_block=gpu_block,
                    cuda_func=cuda_func,
                    cuda_atomic=cuda_atomic,
                    time=time,
                    prog_bar=grid_prog_bar,
                    try_sparse=try_sparse)

            except BaseException:
                self.map_error.append(mol)
                self.logger.exception(
                    f'Error during the mapping of {mol}')

        # remove the molecule with issues
        if self.map_error:
            if remove_error:
                for mol in self.map_error:
                    del f5[mol]
                self.logger.warning(
                    f"Molecules with errored feature mapping are removed:\n"
                    f"{self.map_error}")
            else:
                self.logger.warning(
                    f"The following moleclues have errored feature mapping:\n"
                    f"{self.map_error}")

        # close he hdf5 file
        f5.close()

# ====================================================================================
#
#       REMOVE DATA FROM THE DATA SET
#
# ====================================================================================

    def remove(self, feature=True, pdb=True, points=True, grid=False):
        """Remove data from the data set.

        Equivalent to the cleandata command line tool. Once the data has been
        removed from the file it is impossible to add new features/targets

        Args:
            feature (bool, optional): Remove the features
            pdb (bool, optional): Remove the pdbs
            points (bool, optional): remove teh grid points
            grid (bool, optional): remove the maps
        """

        self.logger.debug('Remove features')

        # name of the hdf5 file
        f5 = h5py.File(self.hdf5, 'a')

        # get the folder names
        mol_names = f5.keys()

        for name in mol_names:

            mol_grp = f5[name]

            if feature and 'features' in mol_grp:
                del mol_grp['features']
                del mol_grp['features_raw']
            if pdb and 'pdb' in mol_grp:
                del mol_grp['pdb']
            if points and 'grid_points' in mol_grp:
                del mol_grp['grid_points']
            if grid and 'mapped_features' in mol_grp:
                del mol_grp['mapped_features']

        f5.close()

        # reclaim the space
        os.system('h5repack %s _tmp.h5py' % self.hdf5)
        os.system('mv _tmp.h5py %s' % self.hdf5)


# ====================================================================================
#
#       Simply tune or test the kernel
#
# ====================================================================================


    def _tune_cuda_kernel(self, grid_info, cuda_kernel='kernel_map.c', func='gaussian'):  # pragma: no cover
        """Tune the CUDA kernel using the kernel tuner
        http://benvanwerkhoven.github.io/kernel_tuner/

        Args:
            grid_info (dict): information for the grid definition
            cuda_kernel (str, optional): file containing the kernel
            func (str, optional): function in the kernel to be used

        Raises:
            ValueError: If the tuner has not been used
        """

        try:
            from kernel_tuner import tune_kernel
        except BaseException:
            print(
                'Install the Kernel Tuner: \n \t\t pip install kernel_tuner')
            print('http://benvanwerkhoven.github.io/kernel_tuner/')

        # fills in the grid data if not provided: default = NONE
        grinfo = ['number_of_points', 'resolution']
        for gr in grinfo:
            if gr not in grid_info:
                raise ValueError(
                    '%s must be specified to tune the kernel')

        # define the grid
        nx, ny, nz = grid_info['number_of_points']
        dx, dy, dz = grid_info['resolution']
        lx, ly, lz = nx * dx, ny * dy, nz * dz

        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        z = np.linspace(0, lz, nz)

        # create the dictionary containing the tune parameters
        tune_params = OrderedDict()
        tune_params['block_size_x'] = [2, 4, 8, 16, 32]
        tune_params['block_size_y'] = [2, 4, 8, 16, 32]
        tune_params['block_size_z'] = [2, 4, 8, 16, 32]

        # define the final grid
        grid = np.zeros(grid_info['number_of_points'])

        # arguments of the CUDA function
        x0, y0, z0 = np.float32(0), np.float32(0), np.float32(0)
        alpha = np.float32(0)
        args = [alpha, x0, y0, z0, x, y, z, grid]

        # dimensionality
        problem_size = grid_info['number_of_points']

        # get the kernel
        kernel = os.path.dirname(
            os.path.abspath(__file__)) + '/' + cuda_kernel
        kernel_code_template = open(kernel, 'r').read()

        npts = grid_info['number_of_points']
        res = grid_info['resolution']
        kernel_code = kernel_code_template % {
            'nx': npts[0], 'ny': npts[1], 'nz': npts[2], 'RES': np.max(res)}
        tunable_kernel = self._tunable_kernel(kernel_code)

        # tune
        tune_kernel(func, tunable_kernel,
                    problem_size, args, tune_params)


# ====================================================================================
#
#       Simply test the kernel
#
# ====================================================================================


    def _test_cuda(self, grid_info, gpu_block=8, cuda_kernel='kernel_map.c', func='gaussian'):  # pragma: no cover
        """Test the CUDA kernel.

        Args:
            grid_info (dict): Information for the grid definition
            gpu_block (int, optional): GPU block size to be used
            cuda_kernel (str, optional): File containing the kernel
            func (str, optional): function in the kernel to be used

        Raises:
            ValueError: If the kernel has not been installed
        """

        from time import time

        # fills in the grid data if not provided: default = NONE
        grinfo = ['number_of_points', 'resolution']
        for gr in grinfo:
            if gr not in grid_info:
                raise ValueError(
                    '%s must be specified to tune the kernel')

        # get the cuda function
        npts = grid_info['number_of_points']
        res = grid_info['resolution']
        module = self._compile_cuda_kernel(cuda_kernel, npts, res)
        cuda_func = self._get_cuda_function(module, func)

        # define the grid
        nx, ny, nz = grid_info['number_of_points']
        dx, dy, dz = grid_info['resolution']
        lx, ly, lz = nx * dx, ny * dy, nz * dz

        # create the coordinate
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        z = np.linspace(0, lz, nz)

        # book memp on the gpu
        x_gpu = gpuarray.to_gpu(x.astype(np.float32))
        y_gpu = gpuarray.to_gpu(y.astype(np.float32))
        z_gpu = gpuarray.to_gpu(z.astype(np.float32))
        grid_gpu = gpuarray.zeros(
            grid_info['number_of_points'], np.float32)

        #  make sure we have three block value
        if not isinstance(gpu_block, list):
            gpu_block = [gpu_block] * 3

        #  get the grid
        gpu_grid = [int(np.ceil(n / b))
                    for b, n in zip(gpu_block, grid_info['number_of_points'])]
        print('GPU BLOCK:', gpu_block)
        print('GPU GRID :', gpu_grid)

        xyz_center = np.random.rand(500, 3).astype(np.float32)
        alpha = np.float32(1)
        t0 = time()
        for xyz in xyz_center:
            x0, y0, z0 = xyz
            cuda_func(alpha, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu,
                      block=tuple(gpu_block), grid=tuple(gpu_grid))

        print('Done in: %f ms' % ((time() - t0) * 1000))

# ====================================================================================
#
#       Routines needed to handle CUDA
#
# ====================================================================================

    @staticmethod
    def _compile_cuda_kernel(cuda_kernel, npts, res):  # pragma: no cover
        """Compile the cuda kernel.

        Args:
            cuda_kernel (str): filename
            npts (tuple(int)): number of grid points in each direction
            res (tuple(float)): resolution in each direction

        Returns:
            compiler.SourceModule: compiled kernel
        """
        # get the cuda kernel path
        kernel = os.path.dirname(
            os.path.abspath(__file__)) + '/' + cuda_kernel
        kernel_code_template = open(kernel, 'r').read()
        kernel_code = kernel_code_template % {
            'nx': npts[0], 'ny': npts[1], 'nz': npts[2], 'RES': np.max(res)}

        # compile the kernel
        mod = compiler.SourceModule(kernel_code)
        return mod

    @staticmethod
    def _get_cuda_function(module, func_name):  # pragma: no cover
        """Get a single function from the compiled kernel.

        Args:
            module (compiler.SourceModule): compiled kernel module
            func_name (str): Name of the funtion

        Returns:
            func: cuda function
        """
        cuda_func = module.get_function(func_name)
        return cuda_func

    # tranform the kernel to a tunable one
    @staticmethod
    def _tunable_kernel(kernel):  # pragma: no cover
        """Make a tunale kernel.

        Args:
            kernel (str): String of the kernel

        Returns:
            TYPE: tunable kernel
        """
        switch_name = {
            'blockDim.x': 'block_size_x',
            'blockDim.y': 'block_size_y',
            'blockDim.z': 'block_size_z'}
        for old, new in switch_name.items():
            kernel = kernel.replace(old, new)
        return kernel


# ====================================================================================
#
#       FILTER DATASET
#
# ===================================================================================


    def _filter_cplx(self):
        """Filter the name of the complexes."""

        # read the class ID
        with open(self.pdb_select) as f:
            pdb_name = f.readlines()
        pdb_name = [name.split()[0] + '.pdb' for name in pdb_name]

        # create the filters
        tmp_path = []
        for name in pdb_name:
            tmp_path += list(filter(lambda x: name in x,
                                    self.pdb_path))

        # update the pdb_path
        self.pdb_path = tmp_path


# ====================================================================================
#
#       FEATURES ROUTINES
#
# ====================================================================================

    @staticmethod
    def _compute_features(feat_list, pdb_data, featgrp, featgrp_raw, mutant, logger):
        """Compute the features.

        Args:
            feat_list (list(str)): list of function name, e.g.,
                ['deeprank.features.ResidueDensity',
                'deeprank.features.PSSM_IC']
            pdb_data (bytes): PDB translated in bytes
            featgrp (str): name of the group where to store the xyz feature
            featgrp_raw (str): name of the group where to store the raw feature
            mutant (PdbMutantSelection): the selected mutant
            logger (logger): name of logger object

        Return:
            bool: error happened or not
        """
        error_flag = False  # when False: success; when True: failed
        for feat in feat_list:
            try:
                feat_module = importlib.import_module(feat, package=None)
                feat_module.__compute_feature__(pdb_data, featgrp, featgrp_raw, mutant)
            except Exception as ex:
                logger.exception(ex)
                error_flag = True

        return error_flag


# ====================================================================================
#
#       TARGETS ROUTINES
#
# ====================================================================================

    @staticmethod
    def _compute_targets(targ_list, pdb_data, targrp):
        """Compute the targets.

        Args:
            targ_list (list(str)): list of function name
            pdb_data (bytes): PDB translated in btes
            targrp (str): name of the group where to store the targets
            logger (logger): name of logger object
        """
        for targ in targ_list:
            targ_module = importlib.import_module(targ, package=None)
            targ_module.__compute_target__(pdb_data, targrp)


# ====================================================================================
#
#       ADD PDB FILE
#
# ====================================================================================


    def _add_pdb(self, molgrp, pdbfile, name):
        """Add a pdb to a molgrp.

        Args:
            molgrp (str): mopl group where tp add the pdb
            pdbfile (str): psb file to add
            name (str): dataset name in the hdf5 molgroup
        """

        # read the pdb and extract the ATOM lines
        with open(pdbfile, 'rt') as fi:
            data = [line.split('\n')[0]
                    for line in fi if line.startswith('ATOM')]

        #  PDB default line length is 80
        #  http://www.wwpdb.org/documentation/file-format
        data = np.array(data).astype('|S78')
        molgrp.create_dataset(name, data=data)


    def _add_pssm(self, molgrp, pssm_path, name):
        """ Add a pssm to a molgrp

        Args:
            molgrp (str): mopl group where tp add the pdb
            pssm_file (str): pssm file to add
            name (str): dataset name in the hdf5 molgroup
        """

        with open(pssm_path, 'rt') as f:
            lines = [line.strip() for line in f if line.startswith('ATOM')]

        # We expect the PSSM line length not to go over 200
        data = np.array(lines).astype('|S200')
        molgrp.create_dataset(name, data=data)

    # @staticmethod
    def _get_aligned_sqldb(self, pdbfile, dict_align):
        """return a sqldb of the pdb that is aligned as specified in the dict

        Arguments:
            pdbfile {str} -- path ot the pdb
            dict_align {dict} -- dictionanry of options to align the pdb
        """
        if 'selection' not in dict_align.keys():
            dict_align['selection'] = {}

        if 'export' not in dict_align.keys():
            dict_align['export'] = False

        if dict_align['selection'] == 'interface':
            raise ValueError("interface alignment is not supported")

        else:

            sqldb = align_along_axis(pdbfile, axis=dict_align['axis'],
                                     export=dict_align['export'],
                                     **dict_align['selection'])

        return sqldb

# ====================================================================================
#
#       AUGMENTED DATA
#
# ====================================================================================

    @staticmethod
    def _get_aligned_rotation_axis_angle(random_seed, dict_align):
        """Returns the axis and angle of rotation for data
           augmentation with aligned complexes

        Arguments:
            random_seed {int} -- random seed of rotation
            dict_align {dict} -- the dict describing the alignement

        Returns:
            list(float): axis of rotation
            float: angle of rotation
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        angle = 2 * np.pi * np.random.rand()

        if 'plane' in dict_align.keys():
            if dict_align['plane'] == 'xy':
                axis = [0., 0., 1.]
            elif dict_align['plane'] == 'xz':
                axis = [0., 1., 0.]
            elif dict_align['plane'] == 'yz':
                axis = [1., 0., 0.]
            else:
                raise ValueError("plane must be xy, xz or yz")

        elif 'axis' in dict_align.keys():
            if dict_align['axis'] == 'x':
                axis = [1., 0., 0.]
            elif dict_align['axis'] == 'y':
                axis = [0., 1., 0.]
            elif dict_align['axis'] == 'z':
                axis = [0., 0., 1.]
            else:
                raise ValueError("axis must be x, y or z")
        else:
            raise ValueError('dict_align must contains plane or axis')

        return axis, angle

    # add a rotated pdb structure to the database
    def _add_aug_pdb(self, molgrp, pdbfile, name, axis, angle):
        """Add augmented pdbs to the dataset.

        Args:
            molgrp (str): name of the molgroup
            pdbfile (str): pdb file name
            name (str): name of the dataset
            axis (list(float)): axis of rotation
            angle (float): angle of rotation
            dict_align (dict): dict for alignement of the original pdb

        Returns:
            list(float): center of the molecule
        """
        # create the sqldb and extract positions
        sqldb = pdb2sql.pdb2sql(pdbfile)

        # rotate the positions
        pdb2sql.transform.rot_axis(sqldb, axis, angle)

        # get molecule center
        xyz = sqldb.get('x,y,z')
        center = np.mean(xyz, 0)

        # get the pdb-format data
        data = sqldb.sql2pdb()
        data = np.array(data).astype('|S78')
        molgrp.create_dataset(name, data=data)

        # close the db
        sqldb._close()

        return center

    # rotate th xyz-formatted feature in the database

    @staticmethod
    def _rotate_feature(molgrp, axis, angle, center, feat_name='all'):
        """Rotate the raw feature values.

        Args:
            molgrp (str): name pf the molgrp
            axis (list(float)): axis of rotation
            angle (float): angle of rotation
            center (list(float)): center of rotation
            feat_name (str): name of the feature to rotate or 'all'
        """
        if feat_name == 'all':
            feat = list(molgrp['features'].keys())
        else:
            feat = feat_name
            if not isinstance(feat, list):
                feat = list(feat)

        for fn in feat:

            # extract the data
            data = molgrp['features/' + fn][()]

            # if data not empty
            if data.shape[0] != 0:

                # xyz
                xyz = data[:, 1:4]

                # get rotated xyz
                xyz_rot = pdb2sql.transform.rot_xyz_around_axis(
                    xyz, axis, angle, center)

                # put back the data
                molgrp['features/' + fn][:, 1:4] = xyz_rot
