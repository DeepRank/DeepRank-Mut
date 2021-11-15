import numpy
import torch
import torch.cuda
import logging
import os

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
                 os.path.join(pdb_root, pdb_ac.lower()[1: 3], "pdb{}.ent".format(pdb_ac.lower()))]:

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

    # This is the list of atom objects, that will be returned.
    atoms = []

    # Iterate over the atom output from pdb2sql
    request_s = "x,y,z,rowID,name,element,chainID,resSeq,resName,iCode"
    for row in pdb2sql.get(request_s):

        try:
            x, y, z, atom_number, atom_name, element, chain_id, residue_number, residue_name, insertion_code = row
        except:
            raise ValueError("Got unexpected row {} for {}".format(row, request_s))

        # We use None as the default insertion code
        if insertion_code == "":
            insertion_code = None

        # Make sure that the residue is in the working directory:
        residue_id = (chain_id, residue_number, insertion_code)
        if residue_id not in residues:
            residues[residue_id] = Residue(int(residue_number), residue_name, chain_id, insertion_code)

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        # Create the atom object and link it to the residue:
        atom = Atom(atom_number, atom_position, chain_id, atom_name, element, residues[residue_id])
        residues[residue_id].atoms.append(atom)
        atoms.append(atom)

    return atoms


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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # get all the atoms in the pdb file:
    atoms = get_atoms(pdb2sql)
    count_atoms = len(atoms)
    atom_positions = torch.tensor([atom.position for atom in atoms]).to(device)
    atoms_in_residue = torch.tensor([atom.residue.number == residue_number and
                                     atom.chain_id == chain_id and
                                     atom.residue.insertion_code == insertion_code for atom in atoms]).to(device)

    # calculate euclidean distances
    atom_distance_matrix = torch.cdist(atom_positions, atom_positions, p=2)

    # select pairs that are close enough
    neighbour_matrix = atom_distance_matrix < max_interatomic_distance

    # select pairs of which only one of the atoms is from the residue
    atoms_in_residue_matrix = atoms_in_residue.expand(count_atoms, count_atoms)
    atoms_in_residue_matrix = torch.logical_xor(atoms_in_residue_matrix,
                                                atoms_in_residue_matrix.transpose(0, 1))
    residue_neighbour_matrix = torch.logical_and(atoms_in_residue_matrix, neighbour_matrix)

    # Create a set of pair objects
    neighbour_pairs = set([])
    for index0, index1 in torch.nonzero(residue_neighbour_matrix):
        neighbour_pairs.add((Pair(atoms[index0], atoms[index1])))

    return neighbour_pairs
