import tempfile
import os

import h5py
import numpy

from deeprank.domain.amino_acid import alanine, serine
from deeprank.features.dbnsfp import __compute_feature__, REVELSCORE_FEATURE_NAME
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection


def test_dbnsfp():

    environment = Environment(pdb_root="test/data/pdb",
                              dbnsfp_path="test/data/test_dbNSFP.hdf5")
    variant = PdbVariantSelection(pdb_ac="A6NGL4", chain_id="A",
                                  residue_number=90,
                                  wildtype_amino_acid=alanine, variant_amino_acid=serine,
                                  enst_accession='ENST00000383000',
                                  protein_accession='A6NGL4',
                                  protein_residue_number=90)

    tmp_file, tmp_path = tempfile.mkstemp()
    os.close(tmp_file)

    with h5py.File(tmp_path, 'w') as f5:

        test_group = f5.require_group("test")

        __compute_feature__(environment, 0.0, test_group, variant)

        data = test_group[REVELSCORE_FEATURE_NAME][()]

        assert numpy.any(data == 0.025)

    os.remove(tmp_path)
