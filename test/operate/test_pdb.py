import numpy
import pkg_resources
import os

from pdb2sql import pdb2sql
from nose.tools import ok_

from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_atoms, is_xray


def test_xray():
    for path in ["test/data/pdb/101M/101M.pdb", "test/data/pdb/1CRN/1CRN.pdb", "test/data/pdb/1MEY/1MEY.pdb"]:
        with open(path, 'rt') as f:
            assert is_xray(f), "{} is not identified as x-ray".format(path)

    with open("test/data/pdb/1A6B/1a6b.pdb", 'rt') as f:
        assert not is_xray(f), "1a6b was identified as x-ray"


def test_altloc_5mnh():
    "Test with a structure where residue 153 altloc C has highest occupancy"

    pdb_path = "test/data/pdb/5MNH/5MNH.pdb"

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

    finally:
        pdb._close()

    selection = [atom for atom in atoms if atom.residue.number == 153 and
                                           atom.chain_id == "A" and
                                           atom.name == "CA"]

    assert len(selection) == 1, "got {} of the same atom".format(len(selection))

    assert selection[0].altloc == 'C', "got atom {} instead".format(selection[0].altloc)

def test_altloc_5eyu():
    "Test with a structure where residue 8 has two altlocs with equal occupancy"

    pdb_path = "test/data/pdb/5EYU/5EYU.pdb"

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

    finally:
        pdb._close()

    selection = [atom for atom in atoms if atom.residue.number == 8 and
                                           atom.chain_id == "A" and
                                           atom.name == "CA"]

    assert len(selection) == 1, "got {} of the same atom".format(len(selection))


def test_get_atoms():
    pdb_path = "test/data/pdb/101M/101M.pdb"

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

        ok_(len(atoms) > 0)
    finally:
        pdb._close()


def test_nmr():
    pdb_path = "test/data/pdb/1CR4/1CR4.pdb"

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

        ok_(len(atoms) > 0)

        ok_(atoms[0].position.shape == (3,))

    finally:
        pdb._close()


def _find_atom(atoms, chain_id, residue_number, name):
    matching_atoms = [atom for atom in atoms if atom.chain_id == chain_id and
                                                atom.name == name and atom.residue.number == residue_number]

    assert len(matching_atoms) == 1, "Expected exacly one matching atom, got {}".format(len(matching_atoms))

    return matching_atoms[0]


def _find_residue(atoms, chain_id, residue_number):
    matching_atoms = [atom for atom in atoms if atom.chain_id == chain_id and
                                                atom.residue.number == residue_number]

    assert len(matching_atoms) > 0, "Expected at least one matching atom, got zero"

    return matching_atoms[0].residue


def test_residue_contact_atoms():

    pdb_path = "test/1AK4/native/1AK4.pdb"

    chain_id = 'D'
    residue_number = 145

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)
        query_residue = _find_residue(atoms, chain_id, residue_number)

        contact_atom_pairs = get_residue_contact_atom_pairs(pdb, chain_id, residue_number, None, 8.5)
        assert len(contact_atom_pairs) > 0, "no contacts found"

        # Check for redundancy (we shouldn't see the same set of atoms twice)
        atom_pairs_encountered = []
        for atom1, atom2 in contact_atom_pairs:

            # Check that it was not seen before:
            assert (atom1.id, atom2.id) not in atom_pairs_encountered, \
                "Atomic pair {} - {} encountered twice".format(atom1, atom2)

            # Remember this pair, as well as its reverse:
            atom_pairs_encountered.extend([(atom1.id, atom2.id), (atom2.id, atom1.id)])


        # List all the atoms in the pairs that we found:
        contact_atoms = set([])
        for atom1, atom2 in contact_atom_pairs:
            contact_atoms.add(atom1)
            contact_atoms.add(atom2)

            # Be sure that this atom pair does not pair two atoms of the same residue:
            ok_(atom1.residue != atom2.residue)

            # Be sure that one of the atoms in the pair is from the query residue:
            ok_(atom1.residue == query_residue or atom2.residue == query_residue)
    finally:
        pdb._close()

    # Now, we need to verify that the function "get_residue_contact_atom_pairs" returned the right pairs.
    # We do so by selecting one close residue and one distant residue.

    neighbour = _find_atom(atoms, 'D', 144, 'CA')  # this residue sits right next to the selected residue
    distant = _find_atom(atoms, 'C', 134, 'OE2')  # this residue is very far away

    # Check that the close residue is present in the list and that the distant residue is absent.
    # Also, the function should not pair a residue with itself.

    ok_(neighbour in contact_atoms)
    ok_(distant not in contact_atoms)


def test_contacts_101m():
    pdb_path = "test/data/pdb/101M/101M.pdb"
    chain_id = "A"

    residue_number = 25

    pdb = pdb2sql(pdb_path)
    try:
        contact_atom_pairs = get_residue_contact_atom_pairs(pdb, chain_id, residue_number, None, 10.0)

        assert len(contact_atom_pairs) > 0, "no contacts found"

        for atom1, atom2 in contact_atom_pairs:
            assert atom1.residue is not None
            assert len(atom1.residue.atoms) > 0

            assert atom2.residue is not None
            assert len(atom2.residue.atoms) > 0
    finally:
        pdb._close()
