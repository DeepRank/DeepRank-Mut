import os
import logging

import numpy

from deeprank.models.forcefield.patch import PatchActionType
from deeprank.parse.forcefield.top import TopParser
from deeprank.parse.forcefield.patch import PatchParser
from deeprank.parse.forcefield.residue import ResidueClassParser
from deeprank.parse.forcefield.param import ParamParser
import deeprank.features.forcefield


_log = logging.getLogger(__name__)


_forcefield_directory_path = os.path.dirname(os.path.abspath(deeprank.features.forcefield.__file__))


class AtomicForcefield:
    def __init__(self):
        top_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-5_new.top")
        with open(top_path, 'rt') as f:
            self._top_rows = {(row.residue_name, row.atom_name): row for row in TopParser.parse(f)}

        patch_path = os.path.join(_forcefield_directory_path, "patch.top")
        with open(patch_path, 'rt') as f:
            self._patch_actions = PatchParser.parse(f)

        residue_class_path = os.path.join(_forcefield_directory_path, "residue-classes")
        with open(residue_class_path, 'rt') as f:
            self._residue_class_criteria = ResidueClassParser.parse(f)

        param_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-4_new.param")
        with open(param_path, 'rt') as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _find_matching_residue_class(self, residue):
        for criterium in self._residue_class_criteria:
            if criterium.matches(residue.name, [atom.name for atom in residue.atoms]):
                return criterium.class_name

        return None

    def get_vanderwaals_parameters(self, atom):
        type_ = self._get_type(atom)

        return self._vanderwaals_parameters[type_]

    def _get_type(self, atom):
        atom_name = atom.name
        residue_name = atom.residue.name

        type_ = None

        # check top
        top_key = (residue_name, atom_name)
        if top_key in self._top_rows:
            type_ = self._top_rows[top_key]["type"]

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and \
                        residue_class == action.selection.residue_type and "TYPE" in action:

                    type_ = action["TYPE"]

        if type_ is None:
            raise ValueError("not mentioned in top or patch: {}".format(top_key))

        return type_

    def get_charge(self, atom):
        """
            Args:
                atom(Atom): the atom to get the charge for
            Returns(float): the charge of the given atom
        """

        atom_name = atom.name
        residue_name = atom.residue.name

        charge = None

        # check top
        top_key = (residue_name, atom_name)
        if top_key in self._top_rows:
            charge = float(self._top_rows[top_key]["charge"])

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and \
                        residue_class == action.selection.residue_type:

                    charge = float(action["CHARGE"])

        if charge is None:
            raise ValueError("not mentioned in top or patch: {}".format(top_key))

        return charge

    _EPSILON0 = 1.0
    _COULOMB_CONSTANT = 332.0636

    _VANDERWAALS_DISTANCE_OFF = 10.0
    _VANDERWAALS_DISTANCE_ON = 6.5

    _SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(_VANDERWAALS_DISTANCE_OFF)
    _SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(_VANDERWAALS_DISTANCE_ON)

    def get_vanderwaals_energy(self, atom1, atom2):
        "returns the vanderwaals energy between two atoms"

        distance = get_distance(atom1.position, atom2.position)

        atom1_parameters = self._get_vanderwaals_parameters(atom1)
        atom2_parameters = self._get_vanderwaals_parameters(atom2)

        atom1_epsilon = None
        atom2_epsilon = None
        atom1_sigma = None
        atom2_sigma = None
        if atom1.chain_id != atom2.chain_id:

            atom1_epsilon = atom1_parameters.inter_epsilon
            atom1_sigma = atom1_parameters.inter_sigma
            atom2_epsilon = atom2_parameters.inter_epsilon
            atom2_sigma = atom2_parameters.inter_sigma
        else:
            atom1_epsilon = atom1_parameters.intra_epsilon
            atom1_sigma = atom1_parameters.intra_sigma
            atom2_epsilon = atom2_parameters.intra_epsilon
            atom2_sigma = atom2_parameters.intra_sigma

        average_epsilon = numpy.sqrt(atom1_epsilon * atom2_epsilon)
        average_sigma = 0.5 * (atom1_sigma + atom2_sigma)

        prefactor = None
        if distance > AtomicForcefield._VANDERWAALS_DISTANCE_OFF:
            return 0.0  # too far

        elif distance < AtomicForcefield._VANDERWAALS_DISTANCE_ON:
            prefactor = 1.0  # very close

        else:
            squared_distance = numpy.square(distance)

            prefactor = (pow(AtomicForcefield._SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distance, 2) *
                         (AtomicForcefield._SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distance - 3 * (AtomicForcefield._SQUARED_VANDERWAALS_DISTANCE_ON - squared_distance)) /
                          pow(AtomicForcefield._SQUARED_VANDERWAALS_DISTANCE_OFF - AtomicForcefield._SQUARED_VANDERWAALS_DISTANCE_ON, 3))

        return 4.0 * average_epsilon * (pow(average_sigma / distance, 12) - pow(average_sigma / distance, 6)) * prefactor

    def get_coulomb_energy(self, atom1, atom2):
        "returns the coulomb energy between two atoms"

        distance = get_distance(atom1.position, atom2.position)

        charge1 = self.get_charge(atom1)
        charge2 = self.get_charge(atom2)

        return charge1 * charge2 * AtomicForcefield._COULOMB_CONSTANT / (AtomicForcefield._EPSILON0 * distance)

atomic_forcefield = AtomicForcefield()
