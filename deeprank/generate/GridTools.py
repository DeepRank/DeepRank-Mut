
import itertools
import sys
from time import time

import numpy as np
from scipy.signal import bspline
import pdb2sql

from deeprank.config import logger
from deeprank.tools import sparse
from deeprank.operate.pdb import get_atoms, get_residue_contact_atom_pairs
from deeprank.operate import hdf5data

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


def logif(string, cond): return logger.info(string) if cond else None


class GridTools(object):

    def __init__(self, variant_group, variant,
                 number_of_points=30, resolution=1.,
                 atomic_densities=None, atomic_densities_mode='ind',
                 feature=None, feature_mode='ind',
                 contact_distance=10.0,
                 cuda=False, gpu_block=None, cuda_func=None, cuda_atomic=None,
                 prog_bar=False, time=False, try_sparse=True):
        """Map the feature of a complex on the grid.

        Args:
            variant_group(str): name of the group of the variant in the HDF5 file.
            variant (PdbVariantSelection): The variant
            number_of_points(int, optional): number of points we want in
                each direction of the grid.
            resolution(float, optional): distance(in Angs) between two points.
            atomic_densities(dict, optional): dictionary of element types with
                their vdw radius, see deeprank.config.atom_vdw_radius_noH
            atomic_densities_mode(str, optional): Mode for mapping
                (deprecated must be 'ind').
            feature(None, optional): Name of the features to be mapped.
                By default all the features present in
                hdf5_file['< variant_group > /features/] will be mapped.
            feature_mode(str, optional): Mode for mapping
                (deprecated must be 'ind').
            contact_distance(float, optional): the dmaximum distance
                between two contact atoms default 10.0Å.
            cuda(bool, optional): Use CUDA or not.
            gpu_block(tuple(int), optional): GPU block size to use.
            cuda_func(None, optional): Name of the CUDA function to be
                used for the mapping of the features.
                Must be present in kernel_cuda.c.
            cuda_atomic(None, optional): Name of the CUDA function to be
                used for the mapping of the atomic densities.
                Must be present in kernel_cuda.c.
            prog_bar(bool, optional): print progression bar for
                individual grid (default False).
            time(bool, optional): print timing statistic for
                individual grid (default False).
            try_sparse(bool, optional): Try to store the matrix in
                sparse format (default True).
        """

        # variant and hdf5 file
        self.variant_group = variant_group
        self.variant_basename = variant_group.name

        # variant query
        self.variant = variant

        # hdf5 file to strore data
        self.hdf5 = self.variant_group.file
        self.try_sparse = try_sparse

        # parameter of the grid
        if number_of_points is not None:
            if not isinstance(number_of_points, list):
                number_of_points = [number_of_points] * 3
            self.npts = np.array(number_of_points).astype('int')

        if resolution is not None:
            if not isinstance(resolution, list):
                resolution = [resolution] * 3
            self.res = np.array(resolution)

        # feature requested
        self.atomic_densities = atomic_densities
        self.feature = feature

        # mapping mode
        self.feature_mode = feature_mode
        self.atomic_densities_mode = atomic_densities_mode

        # cuda support
        self.cuda = cuda
        if self.cuda:  # pragma: no cover
            self.gpu_block = gpu_block
            self.gpu_grid = [int(np.ceil(n / b))
                             for b, n in zip(self.gpu_block, self.npts)]

        # cuda
        self.cuda_func = cuda_func
        self.cuda_atomic = cuda_atomic

        # parameter of the atomic system
        self.atom_xyz = None
        self.atom_index = None
        self.atom_type = None

        # grid points
        self.x = None
        self.y = None
        self.z = None

        # grids for calculation of atomic densities
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None

        # dictionaries of atomic densities
        self.atdens = {}

        # conversion from boh to angs for VMD visualization
        self.bohr2ang = 0.52918

        # contact distance to locate the interface
        self.contact_distance = contact_distance

        # progress bar
        self.local_tqdm = lambda x: tqdm(x) if prog_bar else x
        self.time = time

        # if we already have an output containing the grid
        # we update the existing features
        _update_ = False
        if self.variant_basename + '/grid_points/x' in self.hdf5:
            _update_ = True

        if _update_:
            logif(f'\n=Updating grid data for {self.variant_basename}.',
                  self.time)
            self.update_feature()
        else:
            logif(f'\n= Creating grid and grid data for {self.variant_basename}.',
                  self.time)
            self.create_new_data()

    ################################################################

    def create_new_data(self):
        """Create new feature for a given variant."""

        # get the position/atom type .. of the complex
        self.read_pdb()

        # get the contact atoms and interface center
        self.get_contact_center()

        # define the grid
        self.define_grid_points()

        # save the grid points
        self.export_grid_points()

        # map the features
        self.add_all_features()

        # if we wnat the atomic densisties
        self.add_all_atomic_densities()

        # cloe the db file
        self.sqldb._close()

    ################################################################

    def update_feature(self):
        """Update existing feature in a variant."""

        # get the position/atom type .. of the complex
        # get self.sqldb
        self.read_pdb()

        # read the grid from the hdf5
        grid = self.hdf5.get(self.variant_basename + '/grid_points/')
        self.x, self.y, self.z = grid['x'][()], grid['y'][()], grid['z'][()]

        # create the grid
        self.ygrid, self.xgrid, self.zgrid = np.meshgrid(
            self.y, self.x, self.z)

        # set the resolution/dimension
        self.npts = np.array([len(self.x), len(self.y), len(self.z)])
        self.res = np.array(
            [self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]])

        # map the features
        self.add_all_features()

        # if we want the atomic densisties
        self.add_all_atomic_densities()

        # cloe the db file
        self.sqldb._close()

    ################################################################

    def read_pdb(self):
        """Create a sql databse for the pdb."""

        self.sqldb = pdb2sql.interface(self.variant_group.attrs['pdb_path'])

    # get the contact atoms and interface center
    def get_contact_center(self):
        """Get the center of conact atoms."""

        contact_atom_pairs = get_residue_contact_atom_pairs(self.sqldb,
                                                            self.variant.chain_id, self.variant.residue_number,
                                                            self.contact_distance)
        contact_atom_ids = set([])
        for atom1, atom2 in contact_atom_pairs:
            contact_atom_ids.add(atom1.id)
            contact_atom_ids.add(atom2.id)

        # get interface center
        self.center_contact = np.mean(
            np.array(self.sqldb.get('x,y,z', rowID=list(contact_atom_ids))), 0)

    ################################################################
    # shortcut to add all the feature a
    # and atomic densities in just one line
    ################################################################

    # add all the residue features to the data

    def add_all_features(self):
        """Add all the features toa given variant."""

        # map the features
        if self.feature is not None:

            # map the residue features
            dict_data = self.map_features(self.feature)

            # save to hdf5 if specfied
            t0 = time()
            logif('-- Save Features to HDF5', self.time)
            self.hdf5_grid_data(dict_data, 'Feature_%s' % (self.feature_mode))
            logif('      Total %f ms' % ((time() - t0) * 1000), self.time)

    # add all the atomic densities to the data

    def add_all_atomic_densities(self):
        """Add all atomic densities."""

        # if we wnat the atomic densisties
        if self.atomic_densities is not None:

            # compute the atomic densities
            self.map_atomic_densities()

            # save to hdf5
            t0 = time()
            logif('-- Save Atomic Densities to HDF5', self.time)
            self.hdf5_grid_data(self.atdens, 'AtomicDensities_%s' %
                                (self.atomic_densities_mode))
            logif('      Total %f ms' % ((time() - t0) * 1000), self.time)

    ################################################################
    # define the grid points
    # there is an issue maybe with the ordering
    # In order to visualize the data in VMD the Y and X axis must be inverted ...
    # I keep it like that for now as it should not matter for the CNN
    # and maybe we don't need atomic denisties as features
    ################################################################

    def define_grid_points(self):
        """Define the grid points."""

        logif('-- Define %dx%dx%d grid ' %
              (self.npts[0], self.npts[1], self.npts[2]), self.time)
        logif('-- Resolution of %1.2fx%1.2fx%1.2f Angs' %
              (self.res[0], self.res[1], self.res[2]), self.time)

        halfdim = 0.5 * (self.npts * self.res)
        center = self.center_contact

        low_lim = center - halfdim
        hgh_lim = low_lim + self.res * (np.array(self.npts) - 1)

        self.x = np.linspace(low_lim[0], hgh_lim[0], self.npts[0])
        self.y = np.linspace(low_lim[1], hgh_lim[1], self.npts[1])
        self.z = np.linspace(low_lim[2], hgh_lim[2], self.npts[2])

        # there is something fishy about the meshgrid 3d
        # the axis are a bit screwy ....
        # i dont quite get why the ordering is like that
        self.ygrid, self.xgrid, self.zgrid = np.meshgrid(
            self.y, self.x, self.z)

    ################################################################
    # Atomic densities
    # as defined in the paper about ligand in protein
    ################################################################

    # compute all the atomic densities data
    def map_atomic_densities(self, only_contact=True):
        """Map the atomic densities to the grid.

        Args:
            only_contact(bool, optional): Map only the contact atoms

        Raises:
            ImportError: Description
            ValueError: if an unsupported mode is used
        """
        mode = self.atomic_densities_mode
        logif('-- Map atomic densities on %dx%dx%d grid (mode=%s)' %
              (self.npts[0], self.npts[1], self.npts[2], mode), self.time)

        # prepare the cuda memory
        if self.cuda:  # pragma: no cover

            # try to import pycuda
            try:
                from pycuda import driver, compiler, gpuarray, tools
                import pycuda.autoinit
            except BaseException:
                raise ImportError("Error when importing pyCuda in GridTools")

            # book mem on the gpu
            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
            grid_gpu = gpuarray.zeros(self.npts, np.float32)

        # get the contact atoms
        atoms_by_chain = {}
        if only_contact:
            contact_atom_pairs = get_residue_contact_atom_pairs(self.sqldb, self.variant.chain_id, self.variant.residue_number, self.contact_distance)

            for atom1, atom2 in contact_atom_pairs:
                atoms_by_chain[atom1.chain_id] = atoms_by_chain.get(atom1.chain_id, []) + [atom1]
                atoms_by_chain[atom2.chain_id] = atoms_by_chain.get(atom2.chain_id, []) + [atom2]
        else:
            for atom in get_atoms(self.sqldb):
                atoms_by_chain[atom.chain_id] = atoms_by_chain.get(atom.chain_id, []) + [atom]

        # Loop over the atom types:
        for element_type, vdw_rad in self.local_tqdm(self.atomic_densities.items()):

            t0 = time()

            # Loop over the atoms:
            for chain_id, atoms in atoms_by_chain.items():
                if self.cuda:  # if we use CUDA
                    # reset the grid
                    grid_gpu *= 0

                    # get the atomic densities
                    for atom in atoms:
                        if atom.element == element_type:
                            x0, y0, z0 = atom.position.astype(np.float32)
                            vdw = np.float32(vdw_rad)
                            self.cuda_atomic(vdw, x0, y0, z0,
                                             x_gpu, y_gpu, z_gpu, grid_gpu,
                                             block=tuple(self.gpu_block),
                                             grid=tuple(self.gpu_grid))
                            atdens = grid_gpu.get()

                else:  # if we don't use CUDA

                    # init the grid
                    atdens = np.zeros(self.npts)

                    # run on the atoms
                    for atom in atoms:
                        if atom.element == element_type:
                            atdens += self.densgrid(atom.position, vdw_rad)

                if mode == 'ind':
                    key = element_type
                    self.atdens[key] = atdens
                else:
                    raise ValueError('Unsupported atomic density mode {}'.format(mode))

            tgrid = time() - t0
            logif('     Grid    time %f ms' % (tgrid * 1000), self.time)


    # compute the atomic denisties on the grid
    def densgrid(self, center, vdw_radius):
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
        dd = np.sqrt((self.xgrid - x0)**2
                     + (self.ygrid - y0)**2
                     + (self.zgrid - z0)**2)

        dgrid = np.zeros(self.npts)

        index_shortd = dd < vdw_radius
        index_longd = (dd >= vdw_radius) & (dd < 1.5 * vdw_radius)
        dgrid[index_shortd] = np.exp(-2 * dd[index_shortd]**2 / vdw_radius**2)
        dgrid[index_longd] = 4. / np.e**2 / vdw_radius**2 * dd[index_longd]**2 \
            - 12. / np.e**2 / vdw_radius * dd[index_longd] + 9. / np.e**2
        return dgrid

    ################################################################
    # Residue or Atomic features
    # read the file provided in input
    # and map it on the grid
    ################################################################

    @staticmethod
    def _get_feature_row_position_values(row):
        """ Extract metadata from an input xyz feature row.
            The row format is: x y z [values]

            Returns (triple): position(float list of length 3) and values(float list)
        """

        position_dimension = 3

        position = row[:position_dimension]
        values = row[position_dimension:]

        return position, values

    @staticmethod
    def _get_indicative_feature_key(feature_type_name, value_number=None):
        """ Creates a key to be used within the grid feature group.
            The format is: [type name]_[value number]

            Returns (str): the key
        """

        feature_name = feature_type_name
        if value_number is not None:
            feature_name += "_value%03d" % value_number

        return feature_name

    # map residue a feature on the grid
    def map_features(self, featlist, transform=None):
        """Map individual feature to the grid.

        For residue based feature the feature file must be of the format
        chainID residue_name(3-letter)  residue_number [values]

        For atom based feature it must be
        chainID residue_name(3-letter)  residue_number atome_name [values]

        Args:
            featlist (list(str)): list of features to be mapped
            transform (callable, optional): transformation of the feature (?)

        Returns:
            np.array: Mapped features

        Raises:
            ImportError: Description
            ValueError: if an unsupported mode is used
        """

        # declare the total dictionary
        dict_data = {}

        # prepare the cuda memory
        if self.cuda:  # pragma: no cover

            # try to import pycuda
            try:
                from pycuda import driver, compiler, gpuarray, tools
                import pycuda.autoinit
            except BaseException:
                raise ImportError("Error when importing pyCuda in GridTools")

            # book mem on the gpu
            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
            grid_gpu = gpuarray.zeros(self.npts, np.float32)

        # loop over all the features required
        for feature_name in featlist:

            logger.debug('-- Map %s on %dx%dx%d grid ' % (feature_name, self.npts[0], self.npts[1], self.npts[2]))

            # read the data
            featgrp = self.variant_group['features']
            if feature_name in featgrp.keys():
                data = featgrp[feature_name][:]
            else:
                raise ValueError('feature %s not found in the file' % (feature_name))

            # test if the transform is callable
            # and test it on the first line of the data
            # get the data on the first line
            position_dimension = 3

            logger.debug("data shape {}".format(data.shape))

            if data.shape[0] != 0:
                position, data_test = GridTools._get_feature_row_position_values(data[0])

                logger.debug("position={}, data={}".format(position, data_test))

                # define the length of the output
                if transform is None:
                    nFeat = len(data_test)
                elif callable(transform):
                    nFeat = len(transform(data_test))
                else:
                    print('Error transform in map_feature must be callable')
                    return None
            else:
                nFeat = 1

            logger.debug("placing {} features in {}".format(nFeat, feature_name))

            # Initialize the dict that will eventually hold all the data:
            if nFeat == 1:

                fname = GridTools._get_indicative_feature_key(feature_name)
                dict_data[fname] = np.zeros(self.npts)

            else: # do we need that ?!

                for iF in range(nFeat):
                    fname = GridTools._get_indicative_feature_key(feature_name, value_number=iF)
                    dict_data[fname] = np.zeros(self.npts)

            # skip empty features
            if data.shape[0] == 0:
                continue

            # rest the grid and get the x y z values
            if self.cuda:  # pragma: no cover
                grid_gpu *= 0

            # timing
            tprocess = 0
            tgrid = 0

            # map all the features
            for row in self.local_tqdm(data):
                t0 = time()

                # parse the row
                pos, feat_values = GridTools._get_feature_row_position_values(row)

                # postporcess the data
                if callable(transform):
                    feat_values = transform(feat_values)

                # handle the mode
                if self.feature_mode == "diff":
                    raise ValueError("Unsupported feature mode {}".format(self.feature_mode))
                else:
                    coeff = 1.0

                tprocess += time() - t0

                t0 = time()
                # map this feature(s) on the grid(s)
                if not self.cuda:
                    if nFeat == 1:
                        fname = GridTools._get_indicative_feature_key(feature_name)

                        dict_data[fname] += coeff * self.featgrid(pos, feat_values)
                    else:
                        for iF in range(nFeat):
                            fname = GridTools._get_indicative_feature_key(feature_name, iF)

                            dict_data[fname] += coeff * self.featgrid(pos, feat_values[iF])

                # try to use cuda to speed it up
                else:  # pragma: no cover
                    if nFeat == 1:
                        x0, y0, z0 = pos.astype(np.float32)
                        alpha = np.float32(coeff * feat_values)
                        self.cuda_func(alpha,
                                       x0, y0, z0,
                                       x_gpu, y_gpu, z_gpu,
                                       grid_gpu,
                                       block=tuple(self.gpu_block),
                                       grid=tuple(self.gpu_grid))
                    else:
                        raise ValueError('CUDA only possible for single-valued features')

                tgrid += time() - t0

            if self.cuda:  # pragma: no cover
                fname = GridTools._get_indicative_feature_key(feature_name)

                dict_data[fname] = grid_gpu.get()
                driver.Context.synchronize()

            logger.debug('     Process time %f ms' % (tprocess * 1000))
            logger.debug('     Grid    time %f ms' % (tgrid * 1000))
            logger.debug("     Returning data {}" % dict_data)

        return dict_data

    # compute the a given feature on the grid
    def featgrid(self, center, value, type_='fast_gaussian'):
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

        # simple Gaussian
        if type_ == 'gaussian':
            dd = np.sqrt((self.xgrid - x0)**2
                         + (self.ygrid - y0)**2
                         + (self.zgrid - z0)**2)
            dd = value * np.exp(-beta * dd)
            return dd

        # fast gaussian
        elif type_ == 'fast_gaussian':

            cutoff = 5. * beta

            dd = np.sqrt((self.xgrid - x0)**2
                         + (self.ygrid - y0)**2
                         + (self.zgrid - z0)**2)
            dgrid = np.zeros(self.npts)

            dgrid[dd < cutoff] = value * np.exp(-beta * dd[dd < cutoff])

            return dgrid

        # Bsline
        elif type_ == 'bspline':
            spline_order = 4
            spl = bspline((self.xgrid - x0) / self.res[0], spline_order) \
                * bspline((self.ygrid - y0) / self.res[1], spline_order) \
                * bspline((self.zgrid - z0) / self.res[2], spline_order)
            dd = value * spl
            return dd

        # nearest neighbours
        elif type_ == 'nearest':

            # distances
            dx = np.abs(self.x - x0)
            dy = np.abs(self.y - y0)
            dz = np.abs(self.z - z0)

            # index
            indx = np.argsort(dx)[:2]
            indy = np.argsort(dy)[:2]
            indz = np.argsort(dz)[:2]

            # weight
            wx = dx[indx]
            wx /= np.sum(wx)

            wy = dy[indy]
            wy /= np.sum(wy)

            wz = dx[indz]
            wz /= np.sum(wz)

            # define the points
            indexes = [indx, indy, indz]
            points = list(itertools.product(*indexes))

            # define the weight
            W = [wx, wy, wz]
            W = list(itertools.product(*W))
            W = [np.sum(iw) for iw in W]

            # put that on the grid
            dgrid = np.zeros(self.npts)

            for w, pt in zip(W, points):
                dgrid[pt[0], pt[1], pt[2]] = w * value

            return dgrid

        # default
        else:
            raise ValueError(f'Options not recognized for the grid {type_}')

    ################################################################
    # export the grid points for external calculations of some
    # features. For example the electrostatic potential etc ...
    ################################################################

    def export_grid_points(self):
        """export the grid points to the hdf5 file."""

        hdf5data.store_grid_points(self.variant_group, self.x, self.y, self.z)
        hdf5data.store_grid_center(self.variant_group, self.center_contact)

        logger.info("store a grid for {}, centered at {}".format(str(self.variant), self.center_contact))

    # save the data in the hdf5 file

    @staticmethod
    def _check_features(name, features):
        """ Check the feature for values that could cause glitches.

        Args:
            features (np.array): raw feature values
        """

        if np.any(np.isnan(features)):
            raise ValueError("%s: NaN detected" % name)

        if np.any(np.isinf(features)):
            raise ValueError("%s: Infinity detected" % name)


    def hdf5_grid_data(self, dict_data, data_name):
        """Save the mapped feature to the hdf5 file.

        Args:
            dict_data(dict): feature values stored as a dict
            data_name(str): feature name
        """

        hdf5data.store_grid_data(self.variant_group, data_name, dict_data, try_sparse=self.try_sparse)

        for key in dict_data:
            data = np.array(dict_data[key])

            GridTools._check_features("%s[%s] from %s" % (data_name, key, str(self.variant)), data)

            data_summary = "%s<{%f - %f}" % ("x".join([str(n) for n in data.shape]), np.min(data), np.max(data))
            logger.info("stored grid data {} {} for {}: {}\n{}".format(data_name, key, str(self.variant), data_summary, data))

########################################################################
