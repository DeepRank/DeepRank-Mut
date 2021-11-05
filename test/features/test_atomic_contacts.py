import os
import h5py
import tempfile
import shutil

import numpy
from nose.tools import ok_, eq_
from pdb2sql import pdb2sql

from deeprank.features.atomic_contacts import __compute_feature__
from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.forcefield import atomic_forcefield
from deeprank.operate.pdb import get_atoms


def _find_atom(atoms, chain_id, residue_number, atom_name):
    return [atom for atom in atoms if atom.chain_id == chain_id and
                                      atom.residue.number == residue_number and
                                      atom.name == atom_name][0]


def test_forcefield():

    pdb_path = "test/1CRN.pdb"
    variant = PdbVariantSelection(pdb_path, "A", None, None, None)  # don't care about the amino acid change

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


def test_forcefield_on_missing_parameters():

    pdb_path = "test/data/1MEY.pdb"  # this structure has nucleic acid residues

    variant = PdbVariantSelection(pdb_path, "A", None, None, None)  # don't care about the amino acid change

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


def _compute_features(variant):

    tmp_path = tempfile.mkdtemp()
    try:
        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('test')

            features_group = molgrp.require_group('features')
            raw_group = molgrp.require_group('features_raw')

            __compute_feature__(variant.pdb_path, features_group, raw_group, variant)

            vanderwaals_data = features_group['vdwaals'][()]
            coulomb_data = features_group['coulomb'][()]
            charge_data = features_group['charge'][()]

            return charge_data, coulomb_data, vanderwaals_data
    finally:
        shutil.rmtree(tmp_path)


def test_computed_features():

    variants = [PdbVariantSelection("test/1AK4/native/1AK4.pdb", "C", 25, "F", "A"),
                PdbVariantSelection("test/data/1MEY.pdb", "C", 10, "C", "A")]

    for variant in variants:

        charge_data, coulomb_data, vanderwaals_data = _compute_features(variant)

        # Expected: x, y, z, value (=4)
        ok_(vanderwaals_data.size > 0)
        assert(vanderwaals_data.shape[1] == 4), "unexpected vanderwaals shape {}".format(vanderwaals_data.shape)
        ok_(coulomb_data.size > 0)
        assert(coulomb_data.shape[1] == 4), "unexpected coulomb shape {}".format(coulomb_data.shape)
        ok_(charge_data.size > 0)
        assert(charge_data.shape[1] == 4), "unexpected charge shape {}".format(charge_data.shape)


def _create_pdb(contents):

    f, path = tempfile.mkstemp()
    os.close(f)

    with open(path, 'wt') as f:
        f.write(contents)

    return path


def test_physics():

    pdb_path_1 = _create_pdb("""
ATOM      1  OE2 GLU A   1      10.047  10.099   0.625  1.00 13.79           N
ATOM      2  OE2 GLU A   2      16.967  10.784   0.338  1.00 10.80           C
    """) # two negative charges at short distance

    pdb_path_2 = _create_pdb("""
ATOM      1  OE2 GLU A   1       0.047   0.099   0.625  1.00 13.79           N
ATOM      2  CZ  ARG A   2       8.167   0.184   0.038  1.00 10.80           C
    """) # two opposite charges at long distance

    pdb_path_3 = _create_pdb("""
ATOM      1  OE2 GLU A   1      10.047  10.099   0.625  1.00 13.79           N
ATOM      2  CZ  ARG A   2      16.967  10.784   0.338  1.00 10.80           C
    """) # two opposite charges at short distance

    pdb_path_4 = _create_pdb("""
ATOM      1  OE2 GLU A   1       0.047   0.099   0.625  1.00 13.79           N
ATOM      2  OE2 GLU A   2       8.167   0.184   0.038  1.00 10.80           C
    """) # two negative charges at long distance

    try:
        # not caring about the amino acid changes

        variant = PdbVariantSelection(pdb_path_1, "A", 1, None, None)
        _, coulomb_data_1, vanderwaals_data_1 = _compute_features(variant)

        variant = PdbVariantSelection(pdb_path_2, "A", 1, None, None)
        _, coulomb_data_2, vanderwaals_data_2 = _compute_features(variant)

        variant = PdbVariantSelection(pdb_path_3, "A", 1, None, None)
        _, coulomb_data_3, vanderwaals_data_3 = _compute_features(variant)

        variant = PdbVariantSelection(pdb_path_4, "A", 1, None, None)
        _, coulomb_data_4, vanderwaals_data_4 = _compute_features(variant)
    finally:
        os.remove(pdb_path_1)
        os.remove(pdb_path_2)
        os.remove(pdb_path_3)
        os.remove(pdb_path_4)

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
