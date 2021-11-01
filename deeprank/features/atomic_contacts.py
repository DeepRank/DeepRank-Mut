import logging
import pdb2sql
import re
import os

import numpy
import torch
import torch.cuda
from torch_scatter import scatter_sum

from deeprank.config import logger
from deeprank.models.pair import Pair
from deeprank.operate.pdb import get_atoms
from deeprank.features.FeatureClass import FeatureClass
from deeprank.domain.forcefield import atomic_forcefield


EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636

VANDERWAALS_DISTANCE_OFF = 10.0
VANDERWAALS_DISTANCE_ON = 6.5

SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)


COULOMB_FEATURE_NAME = "coulomb"
VANDERWAALS_FEATURE_NAME = "vanderwaals"
CHARGE_FEATURE_NAME = "charge"


def __compute_feature__(pdb_path, feature_group, raw_feature_group, variant):

    feature_object = FeatureClass("Atomic")

    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    max_interatomic_distance = 8.5

    # get the atoms from the pdb
    pdb = pdb2sql(pdb_path)
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    count_atoms = len(atoms)
    atom_positions = torch.tensor([atom.position for atom in atoms]).to(device)
    atoms_in_residue = torch.tensor([atom.residue.number == residue_number and
                                     atom.chain_id == chain_id and
                                     atom.residue.insertion_code == insertion_code for atom in atoms]).to(device)

    # calculate euclidean distances
    atom_distance_matrix = torch.cdist(atom_positions, atom_positions, p=2)

    # select pairs that are close enough
    neighbour_matrix = atom_distance_matrix < max_interatomic_distance

    # select pairs of which only one of the atoms is from the variant residue
    atoms_in_residue_matrix = atoms_in_residue.expand(count_atoms, count_atoms)
    atoms_in_residue_matrix = torch.logical_xor(atoms_in_residue_matrix,
                                                atoms_in_residue_matrix.transpose(0, 1))
    variant_neighbour_matrix = torch.logical_and(atoms_in_residue_matrix, neighbour_matrix)

    # TODO: extend contact to residues

    # initialize the parameters for every pair
    epsilon0_list = []
    epsilon1_list = []
    sigma0_list = []
    sigma1_list = []
    charges0_list = []
    charges1_list = []
    distances_list = []
    all_atoms_involved = set([])
    atom_pair_indices = torch.nonzero(variant_neighbour_matrix)
    for index0, index1 in atom_pair_indices:
        atom0 = atoms[index0]
        atom1 = atoms[index1]

        all_atoms_involved.add(atom0)
        all_atoms_involved.add(atom1)

        vanderwaals_parameters0 = atomic_forcefield.get_vanderwaals_parameters(atom0)
        vanderwaals_parameters1 = atomic_forcefield.get_vanderwaals_parameters(atom1)

        # Either intermolecular or intramolecular
        if atom0.chain_id != atom1.chain_id:

            epsilon0_list.append(vanderwaals_parameters0.inter_epsilon)
            epsilon1_list.append(vanderwaals_parameters1.inter_epsilon)
            sigma0_list.append(vanderwaals_parameters0.inter_sigma)
            sigma1_list.append(vanderwaals_parameters1.inter_sigma)
        else:
            epsilon0_list.append(vanderwaals_parameters0.intra_epsilon)
            epsilon1_list.append(vanderwaals_parameters1.intra_epsilon)
            sigma0_list.append(vanderwaals_parameters0.intra_sigma)
            sigma1_list.append(vanderwaals_parameters1.intra_sigma)

        charges0_list.append(atomic_forcefield.get_charge(atom0))
        charges1_list.append(atomic_forcefield.get_charge(atom1))

        distances_list.append(atom_distance_matrix[index0, index1])

    # convert the parameter lists to tensors
    epsilons0 = torch.tensor(epsilon0_list).to(device)
    epsilons1 = torch.tensor(epsilon1_list).to(device)
    sigmas0 = torch.tensor(sigma0_list).to(device)
    sigmas1 = torch.tensor(sigma1_list).to(device)
    charges0 = torch.tensor(charges0_list).to(device)
    charges1 = torch.tensor(charges1_list).to(device)
    distances = torch.tensor(distances_list).to(device)
    squared_distances = torch.square(distances)
    count_pairs = len(atom_pair_indices)
    count_atoms_involved = len(all_atoms_involved)

    # calculate coulomb potentials
    constant_factor = COULOMB_CONSTANT / EPSILON0

    coulomb_radius_factors = distances * torch.square(torch.ones(count_pairs) - torch.square(distances / max_interatomic_distance))

    coulomb_potentials = charges0 * charges1 * constant_factor * coulomb_radius_factors

    # sum per atom
    coulomb_sums_per_atom_index = scatter_sum(coulomb_potentials, atom_pair_indices[:,0]) + scatter_sum(coulomb_potentials, atom_pair_indices[:,1])

    # calculate vanderwaals potentials
    average_sigmas = 0.5 * (sigmas0 - sigmas1)
    average_epsilons = torch.sqrt(epsilons0 * epsilons1)

    indices_tooclose = (distances < VANDERWAALS_DISTANCE_ON)
    indices_toofar = (distances > VANDERWAALS_DISTANCE_OFF)

    vanderwaals_prefactors = (torch.pow(SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distances, 2) *
                              (SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distances - 3 *
                              (SQUARED_VANDERWAALS_DISTANCE_ON - squared_distances)) /
                              torch.pow(SQUARED_VANDERWAALS_DISTANCE_OFF - SQUARED_VANDERWAALS_DISTANCE_ON, 3))
    vanderwaals_prefactors[indices_tooclose] = 0.0
    vanderwaals_prefactors[indices_toofar] = 1.0

    vanderwaals_potentials = 4.0 * average_epsilons * torch.pow(average_sigmas / distances, 12) - torch.pow(average_sigmas / distances, 6) * vanderwaals_prefactors

    # sum per atom
    vanderwaals_sums_per_atom_index = scatter_sum(vanderwaals_potentials, atom_pair_indices[:,0]) + scatter_sum(vanderwaals_potentials, atom_pair_indices[:,1])
