import logging
import os

import numpy
from scipy.spatial import distance_matrix

from deeprank.models.pair import Pair
from deeprank.models.atom import Atom
from deeprank.models.residue import Residue
from deeprank.config import logger


_log = logging.getLogger(__name__)


def get_pdb_path(pdb_root, pdb_ac):

    for path in [os.path.join(pdb_root, "{}.pdb".format(pdb_ac.lower())),
                 os.path.join(pdb_root, "{}.pdb".format(pdb_ac.upper())),
                 os.path.join(pdb_root, "{}.PDB".format(pdb_ac.upper())),
                 os.path.join(pdb_root, "pdb{}.ent".format(pdb_ac.lower())),
                 os.path.join(pdb_root, pdb_ac.lower()[1: 3], "pdb{}.ent".format(pdb_ac.lower())),
                 os.path.join(pdb_root, pdb_ac.upper(), "{}.pdb".format(pdb_ac.upper())),
                 os.path.join(pdb_root, pdb_ac.upper(), "{}.pdb".format(pdb_ac.lower())),
                 os.path.join(pdb_root, pdb_ac.lower(), "{}.pdb".format(pdb_ac.lower()))]:

        if os.path.isfile(path):
            return path

    raise FileNotFoundError("Cannot find a pdb file for {} in {}".format(pdb_ac, pdb_root))


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False


def get_atoms(pdb2sql):
    """ Builds a list of atom objects, according to the contents of the pdb file.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating

        Returns ([Atom]): all the atoms in the pdb file.
    """

    # This is a working dictionary of residues, identified by their chains and numbers.
    residues = {}

    # This is the dictionary of atom objects, its values will be returned.
    atoms = {}

    # Iterate over the atom output from pdb2sql, select atoms with highest occupancy.
    request_s = "x,y,z,rowID,name,element,chainID,resSeq,resName,iCode,altLoc,occ"
    highest_occupancies = {}
    for row in pdb2sql.get(request_s):

        try:
            x, y, z, atom_number, atom_name, element, chain_id, residue_number, residue_name, insertion_code, altloc, occ = row
        except:
            raise ValueError("Got unexpected row {} for {}".format(row, request_s))

        atom_id = (chain_id, residue_number, insertion_code, atom_name)

        if insertion_code == "":
            insertion_code = None  # default value

        # Make sure that the residue is in the working directory:
        residue_id = (chain_id, residue_number, insertion_code)
        if residue_id not in residues:
            residues[residue_id] = Residue(int(residue_number), residue_name, chain_id, insertion_code)

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        atom_id = (chain_id, residue_number, insertion_code, atom_name)

        # If the occupancy is lower than the previous atom with the same id, skip the atom:
        if atom_id in highest_occupancies:
            highest_occ = highest_occupancies[atom_id]
            if occ <= highest_occ:
                continue

        # otherwise, overwrite..
        atoms[atom_id] = Atom(atom_number, atom_position, chain_id, atom_name, element, residues[residue_id], altloc, occ)
        highest_occupancies[atom_id] = occ

    # Link atoms to residues:
    for (chain_id, residue_number, insertion_code, atom_name), atom in atoms.items():
        residue_id = (chain_id, residue_number, insertion_code)
        residues[residue_id].atoms.append(atom)

    return list(atoms.values())


def get_residue_contact_atom_pairs(pdb2sql, chain_id, residue_number, insertion_code, max_interatomic_distance):
    """ Find interatomic contacts around a residue.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating
            chain_id (str): the chain identifier, where the residue is located
            residue_number (int): the residue number of interest within the chain
            insertion_code (str): insertion code of the residue of interest, may be None
            max_interatomic_distance (float): maximum distance between two atoms

        Returns ([Pair(Atom, Atom)]): pairs of atoms that contact each other
    """

    # get all the atoms in the pdb file:
    atoms = get_atoms(pdb2sql)
    atom_positions = numpy.array([atom.position for atom in atoms])
    atoms_in_residue = numpy.array([atom for atom in atoms
                                    if atom.residue.number == residue_number and
                                    atom.chain_id == chain_id and
                                    atom.residue.insertion_code == insertion_code])
    if len(atoms_in_residue) == 0:
        raise ValueError("no atoms found for residue {} {}{}".format(chain_id, residue_number, insertion_code))

    atoms_in_residue_positions = numpy.array([atom.position for atom in atoms_in_residue])

    # calculate euclidean distances
    atom_distance_matrix = distance_matrix(atom_positions, atoms_in_residue_positions)

    # select pairs that are close enough
    neighbour_matrix = atom_distance_matrix < max_interatomic_distance

    # Create a set of pair objects
    neighbour_pairs = set([])
    for index0, index1 in numpy.transpose(numpy.nonzero(neighbour_matrix)):
        atom0 = atoms[index0]
        atom1 = atoms_in_residue[index1]

        if atom0.residue != atom1.residue:
            neighbour_pairs.add((Pair(atom0, atom1)))

    return neighbour_pairs
