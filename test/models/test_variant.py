from nose.tools import eq_, ok_

from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import alanine, valine, glutamine, glycine, methionine


def test_instance():

    pdb_ac = "1AK4"
    chain_id = "A"
    residue_number = 10
    wt_amino_acid = alanine
    var_amino_acid = glutamine

    selection = PdbVariantSelection(pdb_ac, chain_id, residue_number, wt_amino_acid, var_amino_acid)

    eq_(selection.chain_id, chain_id)

    eq_(selection.residue_number, residue_number)

    eq_(selection.pdb_ac, pdb_ac)

    eq_(selection.wild_type_amino_acid, wt_amino_acid)
    eq_(selection.variant_amino_acid, var_amino_acid)


def test_hash():
    variant1 = PdbVariantSelection("101M", "A", 10, valine, glutamine)
    variant2 = PdbVariantSelection("110M.pdb", "A", 25, glycine, methionine)

    dictionary = {variant1: 1, variant2: 2}

    eq_(dictionary[variant1], 1)
    eq_(dictionary[variant2], 2)

    ok_(hash(variant1) != hash(variant2))
