import os
import h5py
import tempfile
import shutil

import numpy
from nose.tools import ok_, eq_

from deeprank.features.atomic_contacts import __compute_feature__, _PhysicsStorage, AtomicContacts
from deeprank.models.variant import PdbVariantSelection


def test_assign_parameters():
    pdb_path = "test/1CRN.pdb"
    forcefield_path = "deeprank/features/forcefield"
    top_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.top')
    param_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.param')
    patch_path = os.path.join(forcefield_path, 'patch.top')

    variant = PdbVariantSelection(pdb_path, "A", None, None)  # don't care about the amino acid change

    with AtomicContacts(variant, top_path, param_path, patch_path) as feature_obj:

        feature_obj._read_top()
        feature_obj._read_param()
        feature_obj._read_patch()
        feature_obj._assign_parameters()

        # Check that glutamate has a negative charge
        charge = feature_obj.sqldb.get("CHARGE", resSeq=23, resName="GLU", name="OE2")[0]
        ok_(charge < 0.0)

        # Check that arginine has a positive charge
        charge = feature_obj.sqldb.get("CHARGE", resSeq=10, resName="ARG", name="CZ")[0]
        ok_(charge > 0.0)

        # Check that the N-terminal CA has a positive charge
        charge = feature_obj.sqldb.get("CHARGE", resSeq=1, resName="THR", name="CA")[0]
        ok_(charge > 0.0)

        # Check that the phenylalanine ring is neutral
        charge = feature_obj.sqldb.get("CHARGE", resSeq=13, resName="PHE", name="CZ")[0]
        eq_(charge, 0.0)

        # Check that the vanderwaals parameters are all within the 0.0 - 4.0 range
        for inter_epsilon, inter_sigma, intra_epsilon, intra_sigma in \
                feature_obj.sqldb.get("inter_epsilon, inter_sigma, intra_epsilon, intra_sigma"):

            min_ = 0.0
            max_ = 4.0

            ok_(inter_epsilon > min_ and inter_epsilon < max_)
            ok_(inter_sigma > min_ and inter_sigma < max_)
            ok_(intra_epsilon > min_ and intra_epsilon < max_)
            ok_(intra_sigma > min_ and intra_sigma < max_)


def test_compute_feature():
    pdb_path = "test/1AK4/native/1AK4.pdb"

    variant = PdbVariantSelection(pdb_path, 'C', 25, 'A')

    tmp_path = tempfile.mkdtemp()
    try:
        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('1AK4')

            features_group = molgrp.require_group('features')
            raw_group = molgrp.require_group('features_raw')

            __compute_feature__(pdb_path, features_group, raw_group, variant)

            vdwaals_data = features_group['vdwaals']
            coulomb_data = features_group['coulomb']
            charge_data = features_group['charge']

            # Expected: x, y, z, value (=4)
            ok_(vdwaals_data.size > 0)
            ok_(vdwaals_data.size % 4 == 0)
            ok_(coulomb_data.size > 0)
            ok_(coulomb_data.size % 4 == 0)
            ok_(charge_data.size > 0)
            ok_(charge_data.size % 4 == 0)

            vdwaals_data_raw = raw_group['vdwaals_raw']
            coulomb_data_raw = raw_group['coulomb_raw']
            charge_data_raw = raw_group['charge_raw']

            ok_(vdwaals_data_raw.size > 0)
            eq_(type(vdwaals_data_raw[0]), numpy.bytes_)
            ok_(coulomb_data_raw.size > 0)
            eq_(type(coulomb_data_raw[0]), numpy.bytes_)
            ok_(charge_data_raw.size > 0)
            eq_(type(charge_data_raw[0]), numpy.bytes_)

    finally:
        shutil.rmtree(tmp_path)


def test_physics():
    eps = 0.3
    sig = 1.0

    vdw_short = _PhysicsStorage.get_vanderwaals_energy(eps, sig, eps, sig, 7.0)
    vdw_long = _PhysicsStorage.get_vanderwaals_energy(eps, sig, eps, sig, 9.0)

    # if distance > sigma, then LJ energy should be negative and less negative at longer distance

    ok_(vdw_short <= 0.0)
    ok_(vdw_long <= 0.0)
    ok_(vdw_short < vdw_long)

    charge1 = -1.0
    charge2 = 1.0
    charge3 = 1.0
    max_dist = 20.0

    # two opposing charges should attract each other
    # so coulomb energy should be negative and less negative at longer distance

    c_short = _PhysicsStorage.get_coulomb_energy(charge1, charge2, 5.0, max_dist)
    c_long = _PhysicsStorage.get_coulomb_energy(charge1, charge2, 10.0, max_dist)

    ok_(c_short < 0.0)
    ok_(c_long < 0.0)
    ok_(c_short < c_long)

    # two opposing charges should repulse each other
    # so coulomb energy should be positive and less positive at longer distance

    c_short = _PhysicsStorage.get_coulomb_energy(charge3, charge2, 5.0, max_dist)
    c_long = _PhysicsStorage.get_coulomb_energy(charge3, charge2, 10.0, max_dist)

    ok_(c_short > 0.0)
    ok_(c_long > 0.0)
    ok_(c_short > c_long)
