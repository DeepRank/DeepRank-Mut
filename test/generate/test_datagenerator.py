from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import *
from deeprank.generate.DataGenerator import DataGenerator


def test_get_grid_center():
    variant = PdbVariantSelection("test/data/101m.pdb", "A", 25, glycine, alanine)
    DataGenerator.get_grid_center(variant)
