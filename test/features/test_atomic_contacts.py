import os
import h5py
import tempfile
import shutil

import numpy
from nose.tools import ok_, eq_
from pdb2sql import pdb2sql

from deeprank.features.atomic_contacts import (__compute_feature__, CHARGE_FEATURE_NAME,
                                               COULOMB_FEATURE_NAME, VANDERWAALS_FEATURE_NAME)
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.forcefield import atomic_forcefield
from deeprank.operate.pdb import get_atoms, get_pdb_path
from deeprank.models.environment import Environment
from deeprank.domain.amino_acid import *


def _find_atom(atoms, chain_id, residue_number, atom_name):
    return [atom for atom in atoms if atom.chain_id == chain_id and
                                      atom.residue.number == residue_number and
                                      atom.name == atom_name][0]


def test_forcefield():

    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant = PdbVariantSelection("1CRN", "A", None, None, None)  # don't care about the amino acid change

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)
    pdb = pdb2sql(pdb_path)
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    # Check that glutamate has a negative charge
    charge = atomic_forcefield.get_charge(_find_atom(atoms, "A", 23, "OE2"))
    ok_(charge < 0.0)

    # Check that arginine has a positive charge
    charge = atomic_forcefield.get_charge(_find_atom(atoms, "A", 10, "CZ"))
    ok_(charge > 0.0)

    # Check that the N-terminal CA has a positive charge
    charge = atomic_forcefield.get_charge(_find_atom(atoms, "A", 1, "CA"))
    ok_(charge > 0.0)

    # Check that the phenylalanine ring is neutral
    charge = atomic_forcefield.get_charge(_find_atom(atoms, "A", 13, "CZ"))
    eq_(charge, 0.0)

    # Check that the vanderwaals parameters are all within the 0.0 ... 4.0 range
    for atom in atoms:
        p = atomic_forcefield.get_vanderwaals_parameters(atom)

        min_ = 0.0
        max_ = 4.0

        ok_(p.inter_epsilon >= min_ and p.inter_epsilon <= max_)
        ok_(p.inter_sigma >= min_ and p.inter_sigma <= max_)
        ok_(p.intra_epsilon >= min_ and p.intra_epsilon <= max_)
        ok_(p.intra_sigma >= min_ and p.intra_sigma <= max_)


def test_has_negative_features():
    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant = PdbVariantSelection("7req", "A", 255, glutamate, aspartate)

    hdf5_file, hdf5_path = tempfile.mkstemp()
    os.close(hdf5_file)

    try:
        with h5py.File(hdf5_path, 'w') as f5:
            group_xyz = f5.require_group("xyz")
            __compute_feature__(environment, 10.0, group_xyz, variant)

            charges = group_xyz[CHARGE_FEATURE_NAME][()]
            vanderwaals = group_xyz[VANDERWAALS_FEATURE_NAME][()]
            coulomb = group_xyz[COULOMB_FEATURE_NAME][()]
    finally:
        os.remove(hdf5_path)

    assert len(numpy.nonzero(charges < 0.0)) > 0, "no negative charges"
    assert len(numpy.nonzero(vanderwaals < 0.0)) > 0, "no negative vanderwaals"
    assert len(numpy.nonzero(coulomb < 0.0)) > 0, "no negative coulomb"

def test_forcefield_on_missing_parameters():

    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    # this structure has nucleic acid residues
    # don't care about the amino acid change
    variant = PdbVariantSelection("1MEY", "A", None, None, None)

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)
    pdb = pdb2sql(pdb_path)
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    # Check that the vanderwaals parameters are all within the 0.0 ... 4.0 range
    # Charges should be between -1.5 ... +1.5
    for atom in atoms:
        p = atomic_forcefield.get_vanderwaals_parameters(atom)

        min_ = 0.0
        max_ = 4.0

        ok_(p.inter_epsilon >= min_ and p.inter_epsilon <= max_)
        ok_(p.inter_sigma >= min_ and p.inter_sigma <= max_)
        ok_(p.intra_epsilon >= min_ and p.intra_epsilon <= max_)
        ok_(p.intra_sigma >= min_ and p.intra_sigma <= max_)

        c = atomic_forcefield.get_charge(atom)
        ok_(c >= -1.5 and c <= 1.5)


def test_forcefield_on_altlocs():
    "test the forcefield on a structure with atomic altlocs"

    pdb_path = "test/data/pdb/5EYU/5EYU.pdb"

    variant = PdbVariantSelection("5EYU", "A", 8, serine, cysteine)

    pdb = pdb2sql(pdb_path)
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    # Check for NaN values
    for atom in atoms:
        p = atomic_forcefield.get_vanderwaals_parameters(atom)
        c = atomic_forcefield.get_charge(atom)

        ok_(not numpy.any(numpy.isnan([p.inter_epsilon, p.inter_sigma, p.intra_epsilon, p.intra_sigma, c])))



def _compute_features(environment, variant):

    tmp_path = tempfile.mkdtemp()
    try:
        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('test')

            features_group = molgrp.require_group('features')

            __compute_feature__(environment, 10.0, features_group, variant)

            vanderwaals_data = features_group['vdwaals'][()]
            coulomb_data = features_group['coulomb'][()]
            charge_data = features_group['charge'][()]

            return charge_data, coulomb_data, vanderwaals_data
    finally:
        shutil.rmtree(tmp_path)


def test_computed_features():

    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variants = [PdbVariantSelection("101M", "A", 25, None, glycine, alanine),
                PdbVariantSelection("1MEY", "C", 10, None, cysteine, alanine)]

    for variant in variants:

        charge_data, coulomb_data, vanderwaals_data = _compute_features(environment, variant)

        # Expected: x, y, z, value (=4)
        ok_(vanderwaals_data.size > 0)
        assert(vanderwaals_data.shape[1] == 4), "unexpected vanderwaals shape {}".format(vanderwaals_data.shape)
        ok_(coulomb_data.size > 0)
        assert(coulomb_data.shape[1] == 4), "unexpected coulomb shape {}".format(coulomb_data.shape)
        ok_(charge_data.size > 0)
        assert(charge_data.shape[1] == 4), "unexpected charge shape {}".format(charge_data.shape)


def _create_pdb(root, ac, contents):

    path = os.path.join(root, "{}.pdb".format(ac))

    with open(path, 'wt') as f:
        f.write(contents)

    return path


def test_physics():

    tmp_dir = tempfile.mkdtemp()

    try:
        environment = Environment(pdb_root=tmp_dir)

        _create_pdb(tmp_dir, "1TST", """
ATOM      1  OE2 GLU A   1      10.047  10.099   0.625  1.00 13.79           N
ATOM      2  OE2 GLU A   2      16.967  10.784   0.338  1.00 10.80           C
        """) # two negative charges at short distance

        _create_pdb(tmp_dir, "2TST", """
ATOM      1  OE2 GLU A   1       0.047   0.099   0.625  1.00 13.79           N
ATOM      2  CZ  ARG A   2       8.167   0.184   0.038  1.00 10.80           C
        """) # two opposite charges at long distance

        _create_pdb(tmp_dir, "3TST", """
ATOM      1  OE2 GLU A   1      10.047  10.099   0.625  1.00 13.79           N
ATOM      2  CZ  ARG A   2      16.967  10.784   0.338  1.00 10.80           C
        """) # two opposite charges at short distance

        _create_pdb(tmp_dir, "4TST", """
ATOM      1  OE2 GLU A   1       0.047   0.099   0.625  1.00 13.79           N
ATOM      2  OE2 GLU A   2       8.167   0.184   0.038  1.00 10.80           C
        """) # two negative charges at long distance

        # not caring about the amino acid changes

        variant = PdbVariantSelection("1TST", "A", 1, None, None)
        _, coulomb_data_1, vanderwaals_data_1 = _compute_features(environment, variant)

        variant = PdbVariantSelection("2TST", "A", 1, None, None)
        _, coulomb_data_2, vanderwaals_data_2 = _compute_features(environment, variant)

        variant = PdbVariantSelection("3TST", "A", 1, None, None)
        _, coulomb_data_3, vanderwaals_data_3 = _compute_features(environment, variant)

        variant = PdbVariantSelection("4TST", "A", 1, None, None)
        _, coulomb_data_4, vanderwaals_data_4 = _compute_features(environment, variant)

    finally:
        shutil.rmtree(tmp_dir)

    eps = 0.3
    sig = 1.0

    vdw_short = vanderwaals_data_1[0,3]
    vdw_long = vanderwaals_data_2[0,3]

    # if distance > sigma, then LJ energy should be negative and less negative at longer distance

    assert(vdw_short < 0.0), "vdw: short {} is not negative".format(vdw_short)
    assert(vdw_long < 0.0), "vdw: long {} is not negative".format(vdw_long)
    assert(vdw_short < vdw_long), "vdw: short {} >= {} long".format(vdw_short, vdw_long)

    charge1 = -1.0
    charge2 = 1.0
    charge3 = 1.0
    max_dist = 20.0

    # two charges of opposite sign should attract each other
    # so coulomb energy should be negative and less negative at longer distance

    c_short = coulomb_data_3[0,3]
    c_long = coulomb_data_2[0,3]

    ok_(c_short < 0.0)
    ok_(c_long < 0.0)
    ok_(c_short < c_long)

    # two charges op the same sign should repulse each other
    # so coulomb energy should be positive and less positive at longer distance

    c_short = coulomb_data_1[0,3]
    c_long = coulomb_data_4[0,3]

    ok_(c_short > 0.0)
    ok_(c_long > 0.0)
    ok_(c_short > c_long)


def test_large_structure():
    environment = Environment(pdb_root="test/data/pdb")

    variant = PdbVariantSelection("2Y69", "A", 145, isoleucine, leucine)

    _compute_features(environment, variant)
