import tempfile
import os

import h5py
import numpy

from deeprank.domain.amino_acid import cysteine, serine
from deeprank.features.gnomad import __compute_feature__, AF_FEATURE_NAME
from deeprank.models.environment import Environment
from deeprank.models.variant import PdbVariantSelection


def test_dbnsfp():

    environment = Environment(pdb_root="test/data/pdb",
                              gnomad_path="test/data/test_gnomAD.hdf5")
    variant = PdbVariantSelection(pdb_ac="1KLR", chain_id="A",
                                  residue_number=94,
                                  wildtype_amino_acid=serine, variant_amino_acid=cysteine,
                                  enst_accession='ENST00000625061',
                                  protein_accession='P08048',
                                  protein_residue_number=94)

    tmp_file, tmp_path = tempfile.mkstemp()
    os.close(tmp_file)

    with h5py.File(tmp_path, 'w') as f5:

        test_group = f5.require_group("test")

        __compute_feature__(environment, 0.0, test_group, variant)

        data = test_group[AF_FEATURE_NAME][()]

        assert numpy.any(data == 0.0)

    os.remove(tmp_path)


def test_linked_dbnsfp():

    environment = Environment(pdb_root="test/data/pdb",
                              gnomad_path="test/data/link_gnomAD.hdf5")
    variant = PdbVariantSelection(pdb_ac="1KLR", chain_id="A",
                                  residue_number=94,
                                  wildtype_amino_acid=serine, variant_amino_acid=cysteine,
                                  enst_accession='ENST00000625061',
                                  protein_accession='P08048',
                                  protein_residue_number=94)

    tmp_file, tmp_path = tempfile.mkstemp()
    os.close(tmp_file)

    with h5py.File(tmp_path, 'w') as f5:

        test_group = f5.require_group("test")

        __compute_feature__(environment, 0.0, test_group, variant)

        data = test_group[AF_FEATURE_NAME][()]

        assert numpy.any(data == 0.0)

    os.remove(tmp_path)
