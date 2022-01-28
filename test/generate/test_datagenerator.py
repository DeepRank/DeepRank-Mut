from deeprank.models.variant import PdbVariantSelection
from deeprank.domain.amino_acid import *
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.models.environment import Environment


def test_get_grid_center():
    environment = Environment(pdb_root="test/data/pdb", device="cpu")

    variant = PdbVariantSelection("101m", "A", 25, glycine, alanine)

    DataGenerator.get_grid_center(environment, variant)
