from typing import List, Union

import h5py
import numpy
from pdb2sql import pdb2sql

from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection
from deeprank.operate.pdb import get_pdb_path



SIZE_DELTA_FEATURE_NAME = "size_delta"
CHARGE_DELTA_FEATURE_NAME = "charge_delta"
POLARITY_DELTA_FEATURE_NAME = "polarity_delta"
HBACCEPTOR_DELTA_FEATURE_NAME = "hb_acceptor_delta"
HBDONOR_DELTA_FEATURE_NAME = "hb_donor_delta"


def _store_feature(feature_group_xyz: h5py.Group,
                   atom_positions: numpy.array,
                   feature_name: str,
                   value: Union[float, int, numpy.array]):
    """
    atom_positions: [n_atom, 3] x,y,z per atom
    value: one float or a vector
    """

    if type(value) == float or type(value) == int:
        values = [value]
    else:
        values = list(value)

    data = [list(position) + values for position in atom_positions]

    feature_group_xyz.create_dataset(feature_name, data=data)


def __compute_feature__(environment: Environment,
                        distance_cutoff: float,
                        feature_group: h5py.Group,
                        variant: PdbVariantSelection):

    "computes features that express the difference in physiochemical properties of the swapped amino acid"

    # get the C-alpha position
    pdb = pdb2sql(get_pdb_path(environment.pdb_root, variant.pdb_ac))
    if variant.insertion_code is not None:
        atom_positions = pdb.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, iCode=variant.insertion_code)
    else:
        atom_positions = pdb.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number)

    wildtype_amino_acid = variant.wildtype_amino_acid
    variant_amino_acid = variant.variant_amino_acid

    size_delta = variant_amino_acid.size - wildtype_amino_acid.size
    _store_feature(feature_group, atom_positions, SIZE_DELTA_FEATURE_NAME, size_delta)

    charge_delta = variant_amino_acid.charge - wildtype_amino_acid.charge
    _store_feature(feature_group, atom_positions, CHARGE_DELTA_FEATURE_NAME, charge_delta)

    polarity_delta = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
    _store_feature(feature_group, atom_positions, POLARITY_DELTA_FEATURE_NAME, polarity_delta)

    hb_acceptor_delta = variant_amino_acid.count_hydrogen_bond_acceptors - wildtype_amino_acid.count_hydrogen_bond_acceptors
    _store_feature(feature_group, atom_positions, HBACCEPTOR_DELTA_FEATURE_NAME, hb_acceptor_delta)

    hb_donor_delta = variant_amino_acid.count_hydrogen_bond_donors - wildtype_amino_acid.count_hydrogen_bond_donors
    _store_feature(feature_group, atom_positions, HBDONOR_DELTA_FEATURE_NAME, hb_donor_delta)

