import os

import numpy

from deeprank.models.variant import PdbVariantSelection
from deeprank.tools import sparse
from deeprank.domain.amino_acid import amino_acids

def get_variant_group_name(variant):
    """
        Args:
            variant (PdbVariantSelection): a variant object
        Returns (str): an unique name for a given variant object
    """

    mol_name = str(variant)

    return "%s-%s" % (mol_name, str(hash(variant)).replace('-', 'm'))


def store_variant(variant_group, variant):
    """ Stores the variant in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            variant (PdbVariantSelection): the variant object
    """

    variant_group.attrs['pdb_ac'] = variant.pdb_ac

    variant_group.attrs['variant_chain_id'] = variant.chain_id

    variant_group.attrs['variant_residue_number'] = variant.residue_number

    if variant.insertion_code is not None:
        variant_group.attrs['variant_insertion_code'] = variant.insertion_code

    variant_group.attrs['variant_amino_acid_name'] = variant.variant_amino_acid.name
    variant_group.attrs['wild_type_amino_acid_name'] = variant.wild_type_amino_acid.name

    if variant.protein_accession is not None and variant.protein_residue_number is not None:
        variant_group.attrs['variant_protein_accession'] = variant.protein_accession
        variant_group.attrs['variant_protein_residue_number'] = variant.protein_residue_number

    if variant.enst_accession is not None:
        variant_group.attrs['variant_enst_accession'] = variant.enst_accession


def load_variant(variant_group):
    """ Loads the variant from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (PdbVariantSelection): the variant object
    """

    pdb_ac = variant_group.attrs['pdb_ac']

    chain_id = str(variant_group.attrs['variant_chain_id'])

    residue_number = int(variant_group.attrs['variant_residue_number'])

    if 'variant_insertion_code' in variant_group.attrs:
        insertion_code = str(variant_group.attrs['variant_insertion_code'])
    else:
        insertion_code = None

    if 'variant_protein_accession' in variant_group.attrs and 'variant_protein_residue_number' in variant_group.attrs:

        protein_accession = str(variant_group.attrs['variant_protein_accession'])
        protein_residue_number = int(variant_group.attrs['variant_protein_residue_number'])
    else:
        protein_accession = None
        protein_residue_number = None

    amino_acids_by_name = {amino_acid.name: amino_acid for amino_acid in amino_acids}

    variant_amino_acid = amino_acids_by_name[variant_group.attrs['variant_amino_acid_name']]
    wild_type_amino_acid = amino_acids_by_name[variant_group.attrs['wild_type_amino_acid_name']]

    if 'variant_enst_accession' in variant_group.attrs:
        enst_accession = variant_group.attrs['variant_enst_accession']
    else:
        enst_accession = None

    variant = PdbVariantSelection(pdb_ac, chain_id, residue_number, wild_type_amino_acid, variant_amino_acid, insertion_code=insertion_code,
                                  protein_accession=protein_accession, protein_residue_number=protein_residue_number,
                                  enst_accession=enst_accession)

    return variant


def store_grid_center(variant_group, center):
    """ Stores the center position in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            center (float, float, float): xyz position of the center
    """

    grid_group = variant_group.require_group("grid_points")

    if 'center' in grid_group:
        del(grid_group['center'])

    grid_group.create_dataset('center', data=center, compression='lzf', chunks=True)


def load_grid_center(variant_group):
    """ Loads the center position from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (float, float, float): xyz position of the center
    """

    grid_group = variant_group['grid_points']

    return numpy.array(grid_group['center'])


def store_grid_points(variant_group, x_coords, y_coords, z_coords):
    """ Stores the grid point coordinates in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            x_coords (list(float)): the x coords of the grid points
            y_coords (list(float)): the y coords of the grid points
            z_coords (list(float)): the z coords of the grid points
    """

    grid_group = variant_group.require_group("grid_points")

    for coord in ['x', 'y', 'z']:
        if coord in grid_group:
            del(grid_group[coord])

    grid_group.create_dataset('x', data=x_coords, compression='lzf', chunks=True)
    grid_group.create_dataset('y', data=y_coords, compression='lzf', chunks=True)
    grid_group.create_dataset('z', data=z_coords, compression='lzf', chunks=True)


def load_grid_points(variant_group):
    """ Loads the grid point coordinates from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (list(float), list(float), list(float)): the x, y and z coordinates of the grid points
    """

    grid_group = variant_group['grid_points']

    x_coords = numpy.array(grid_group['x'])
    y_coords = numpy.array(grid_group['y'])
    z_coords = numpy.array(grid_group['z'])

    return (x_coords, y_coords, z_coords)


def store_grid_data(variant_group, feature_name, feature_dict, try_sparse=True):
    """ Store 3D grid data in a variant group.

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            feature_name (str): the name of the feature to store
            feature_dict (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    feature_group = variant_group.require_group("mapped_features/%s" % feature_name)

    for subfeature_name, subfeature_data in feature_dict.items():

        # Remove the old data (if present).
        if subfeature_name in feature_group:
            del(feature_group[subfeature_name])

        # Create the subfeature group anew.
        subfeature_group = feature_group.create_group(subfeature_name)

        if try_sparse:
            spg = sparse.FLANgrid()
            spg.from_dense(subfeature_data, beta=1E-2)

        if try_sparse and spg.sparse:
            subfeature_group.attrs['sparse'] = True
            subfeature_group.attrs['type'] = 'sparse_matrix'
            subfeature_group.create_dataset('index', data=spg.index, compression='lzf', chunks=True)
            subfeature_group.create_dataset('value', data=spg.value, compression='lzf', chunks=True)
        else:
            subfeature_group.attrs['sparse'] = False
            subfeature_group.attrs['type'] = 'dense_matrix'
            subfeature_group.create_dataset('value', data=subfeature_data, compression='lzf', chunks=True)


def load_grid_data(variant_group, feature_name):
    """ Load 3D grid data from a variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            feature_name (str): the name of the feature to store

        Returns (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    grid_xs, grid_ys, grid_zs = load_grid_points(variant_group)
    grid_shape = (len(grid_xs), len(grid_ys), len(grid_zs))

    feature_group = variant_group["mapped_features/%s" % feature_name]

    grid_data = {}
    for subfeature_name in feature_group.keys():
        subfeature_group = feature_group[subfeature_name]

        if subfeature_group.attrs['sparse']:

            spg = sparse.FLANgrid(True,
                                  subfeature_group['index'][:], subfeature_group['value'][:],
                                  grid_shape)
            grid_data[subfeature_name] = numpy.array(spg.to_dense())
        else:
            grid_data[subfeature_name] = numpy.array(subfeature_group['value'][:])

    return grid_data
