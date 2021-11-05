import os
import logging

import numpy

from deeprank.models.forcefield.param import VanderwaalsParam
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
        top_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-4_new.top")
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
        if type_ is None:
            return VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

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
            _log.warning("not mentioned in top or patch: {}".format(top_key))
            return None

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
            _log.warning("not mentioned in top or patch: {}".format(top_key))
            return 0.0

        return charge

atomic_forcefield = AtomicForcefield()
