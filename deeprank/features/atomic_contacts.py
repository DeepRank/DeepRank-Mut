import logging
from pdb2sql import pdb2sql
import re
import os
from typing import List

import h5py
from scipy.spatial import distance_matrix
import numpy
import torch
import torch.cuda
from torch_scatter import scatter_sum

from deeprank.config import logger
from deeprank.models.pair import Pair
from deeprank.operate.pdb import get_atoms, get_pdb_path, get_residue_contact_atom_pairs
from deeprank.features.FeatureClass import FeatureClass
from deeprank.domain.forcefield import atomic_forcefield
from deeprank.models.variant import PdbVariantSelection
from deeprank.models.environment import Environment
from deeprank.models.atom import Atom


_log = logging.getLogger(__name__)


EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636

VANDERWAALS_DISTANCE_OFF = 10.0
VANDERWAALS_DISTANCE_ON = 6.5

SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)

MAX_BOND_DISTANCE = 2.1

COULOMB_FEATURE_NAME = "coulomb"
VANDERWAALS_FEATURE_NAME = "vdwaals"
CHARGE_FEATURE_NAME = "charge"


def _store_features(feature_group_xyz, feature_name, atoms, values):

    data = [list(atoms[index].position) + [values[index]] for index in range(len(atoms))]

    feature_group_xyz.create_dataset(feature_name, data=data, compression="lzf", chunks=True)

    # We're currently not doing anything with the raw features.


def _get_atoms_around_variant(environment: Environment, variant: PdbVariantSelection) -> List[Atom]:

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    pdb = pdb2sql(pdb_path)

    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(pdb,
                                                           variant.chain_id,
                                                           variant.residue_number,
                                                           variant.insertion_code,
                                                           10.0):

            for atom in (atom1.residue.atoms + atom2.residue.atoms):
                atoms.add(atom)

        atoms = list(atoms)
    finally:
        pdb._close()

    return atoms


def _select_matrix_subset(matrix: torch.Tensor,
                          index_row: torch.Tensor,
                          index_col: torch.Tensor):

    return matrix[index_row][..., index_col]


def __compute_feature__(environment: Environment,
                        max_interatomic_distance: float,
                        feature_group: h5py.Group,
                        variant: PdbVariantSelection):
    """
        For all atoms surrounding the variant, calculate vanderwaals, coulomb and charge features.
        This uses torch for fast computation. The downside of this is that we cannot use python objects.

        Args:
            max_interatomic_distance: max distance (Å) from variant to include atoms
            feature_group: where the features should go
    """

    atoms = _get_atoms_around_variant(environment, variant)

    # Determine which atoms are the variant and which are surroundings.
    variant_indexes = []
    surrounding_indexes = []
    variant_atoms = []
    surrounding_atoms = []
    positions = []
    for index, atom in enumerate(atoms):

        positions.append(atom.position)

        if atom.residue.number == variant.residue_number and \
           atom.chain_id == variant.chain_id and \
           atom.residue.insertion_code == variant.insertion_code:

            variant_indexes.append(index)
            variant_atoms.append(atom)
        else:
            surrounding_atoms.append(atom)
            surrounding_indexes.append(index)
    assert len(variant_indexes) > 0
    assert len(surrounding_indexes) > 0
    variant_indexes = torch.tensor(variant_indexes).to(environment.device)
    surrounding_indexes = torch.tensor(surrounding_indexes).to(environment.device)

    positions = torch.tensor(positions).to(environment.device)
    distance_matrix = torch.cdist(positions, positions, p=2)
    r = _select_matrix_subset(distance_matrix, variant_indexes, surrounding_indexes)

    # get charges
    q = torch.tensor([atomic_forcefield.get_charge(atom)
                      for atom in atoms]).to(environment.device)

    _store_features(feature_group, CHARGE_FEATURE_NAME, atoms, q)

    # calculate coulomb
    cutoff_distance = 8.5
    coulomb_cutoff = torch.square(torch.ones(r.shape).to(environment.device) - torch.square(r / cutoff_distance))
    q1q2 = q[variant_indexes].unsqueeze(dim=1) * q[surrounding_indexes].unsqueeze(dim=0)
    ec = q1q2 * (COULOMB_CONSTANT / EPSILON0) / r * coulomb_cutoff

    #for index0 in range(coulomb_potentials.shape[0]):
    #    for index1 in range(coulomb_potentials.shape[1]):
    #        atom0 = atoms[index0]
    #        atom1 = atoms[index1]
    #
    #        value = coulomb_potentials[index0, index1]
    #
    #        _log.debug(f"{value} for {atom0} - {atom1}")
    #        assert torch.abs(value) < 100.0

    coulomb_per_atom = torch.zeros(len(atoms)).to(environment.device)
    coulomb_per_atom[variant_indexes] = torch.sum(ec, dim=1).float()
    coulomb_per_atom[surrounding_indexes] = torch.sum(ec, dim=0).float()
    _store_features(feature_group, COULOMB_FEATURE_NAME, atoms, coulomb_per_atom)

    # determine whether distances are inter or intra
    bonded_matrix = (distance_matrix < MAX_BOND_DISTANCE).float()
    bonded2_matrix = torch.matmul(bonded_matrix, bonded_matrix)
    bonded3_matrix = torch.matmul(bonded2_matrix, bonded_matrix).bool()  # clamp to 0.0 - 1.0
    intra_matrix = _select_matrix_subset(bonded3_matrix, variant_indexes, surrounding_indexes)
    inter_matrix = _select_matrix_subset(torch.logical_not(bonded3_matrix), variant_indexes, surrounding_indexes)

    # fetch vanderwaals parameters
    inter_epsilon = torch.zeros(len(atoms)).to(environment.device)
    inter_sigma = torch.zeros(len(atoms)).to(environment.device)
    intra_epsilon = torch.zeros(len(atoms)).to(environment.device)
    intra_sigma = torch.zeros(len(atoms)).to(environment.device)
    for index, atom in enumerate(atoms):
        vanderwaals_parameters = atomic_forcefield.get_vanderwaals_parameters(atom)
        inter_epsilon[index] = vanderwaals_parameters.inter_epsilon
        inter_sigma[index] = vanderwaals_parameters.inter_sigma
        intra_epsilon[index] = vanderwaals_parameters.intra_epsilon
        intra_sigma[index] = vanderwaals_parameters.intra_sigma

    # Use intra when less than 3 bonds away from each other.
    sigma = 0.5 * (inter_matrix * (inter_sigma[variant_indexes].unsqueeze(dim=1) + inter_sigma[surrounding_indexes].unsqueeze(dim=0)) +
                   intra_matrix * (intra_sigma[variant_indexes].unsqueeze(dim=1) + intra_sigma[surrounding_indexes].unsqueeze(dim=0)))

    epsilon = torch.sqrt(inter_matrix * inter_epsilon[variant_indexes].unsqueeze(dim=1) * inter_epsilon[surrounding_indexes].unsqueeze(dim=0) +
                         intra_matrix * intra_epsilon[variant_indexes].unsqueeze(dim=1) * intra_epsilon[surrounding_indexes].unsqueeze(dim=0))

    vdw = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    # calculate the cutoff
    r_on = 6.5
    r_off = 8.5

    prefactors = (r_off ** 2 - r ** 2) ** 2 * \
                 (r_off ** 2 - r ** 2 - 3.0 * (r_on ** 2 - r ** 2)) / \
                 (r_off ** 2 - r_on ** 2) ** 3

    prefactors[r > r_off] = 0.0
    prefactors[r < r_on] = 1.0

    vdw *= prefactors

    # ignore bonded atoms
    vdw[r < MAX_BOND_DISTANCE] = 0.0

    # store vanderwaals
    vdw_per_atom = torch.zeros(len(atoms)).to(environment.device)
    vdw_per_atom[variant_indexes] = torch.sum(vdw, dim=1).float()
    vdw_per_atom[surrounding_indexes] = torch.sum(vdw, dim=0).float()
    _store_features(feature_group, VANDERWAALS_FEATURE_NAME, atoms, vdw_per_atom)

    #for index0 in range(vdw.shape[0]):
    #    for index1 in range(vdw.shape[1]):
    #        atom0 = variant_atoms[index0]
    #        atom1 = surrounding_atoms[index1]
    #
    #        value = vdw[index0, index1]
    #
    #        _log.info(f"distance {r[index0, index1]}")
    #        _log.info(f"intra {intra_matrix[index0, index1]}, inter {inter_matrix[index0, index1]}")
    #        _log.info(f"epsilon {epsilon[index0, index1]}, sigma {sigma[index0, index1]}")
    #        _log.info(f"prefactor {prefactors[index0, index1]}")
    #        _log.info(f"{value} for {atom0} - {atom1}")
    #        assert torch.abs(value) < 100.0

