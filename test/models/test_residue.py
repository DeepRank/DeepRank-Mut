from nose.tools import ok_

from deeprank.models.residue import Residue


def test_compare():
    residue1 = Residue(1, 'ALA', 'A')
    residue2 = Residue(1, 'ALA', 'A')

    d = {residue1: 0}

    ok_(residue1 in d)
