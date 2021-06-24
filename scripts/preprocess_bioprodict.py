import logging
import sys
import os
import gzip
from argparse import ArgumentParser
from threading import Thread
from math import isnan
from glob import glob
import traceback

import numpy
import pandas
import csv
from Bio import SeqIO
from mpi4py import MPI
import h5py
from pdb2sql import pdb2sql
import gzip
from pdbecif.mmcif_io import CifFileReader

from deeprank.operate import hdf5data
from deeprank.generate import DataGenerator
from deeprank.models.variant import PdbVariantSelection, VariantClass
from deeprank.config.chemicals import AA_codes_3to1


arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the variant parquet file")
arg_parser.add_argument("map_path", help="the path to the mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("pssm_root", help="the path to the pssm root directory")
arg_parser.add_argument("out_path", help="the path to the output hdf5 file")


args = arg_parser.parse_args()


logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.DEBUG)
_log = logging.getLogger(__name__)


mpi_comm = MPI.COMM_WORLD

feature_modules = ["deeprank.features.atomic_contacts",
                   "deeprank.features.accessibility",
                   "deeprank.features.neighbour_profile"]
target_modules = ["deeprank.targets.variant_class"]

grid_info = {
   'number_of_points': [30,30,30],
   'resolution': [1.,1.,1.],
   'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
}


def preprocess(variants, hdf5_path):
    data_generator = DataGenerator(variants,
                                   compute_features=feature_modules,
                                   compute_targets=target_modules,
                                   hdf5=hdf5_path, mpi_comm=mpi_comm)
    data_generator.create_database()
    data_generator.map_features(grid_info)


def get_pssm_paths(pssm_root, pdb_ac):
    paths = glob(os.path.join(pssm_root, "%s/pssm/%s.?.pdb.pssm" % (pdb_ac.lower(), pdb_ac.lower())))
    return {path.split('.')[1]: path for path in paths}


# in conservation table:
# needed:  pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
# present:  amino_acid  sub_sequencecount  sub_consv_A  sub_consv_B  sub_consv_C  sub_consv_D  ...  sub_consv_V  sub_consv_W  sub_consv_X  sub_consv_Y  sub_consv_Z  sub_consv_gap


def get_variant_data(parq_path, hdf5_path, pdb_root, pssm_root):

    class_table = pandas.read_parquet(parq_path)
    variant_table = pandas.read_hdf(hdf5_path, "variants")
    conservation_table = pandas.read_hdf(hdf5_path, "conservation")
    pdb_table = pandas.read_hdf(hdf5_path, "pdbs")

    objects = set([])

    # Iterate over all variants in the parq file:
    for variant, variant_class in class_table['class'].items():

        variant = variant.split('.')[1]
        enst_ac = variant[:15]
        swap = variant[15:]

        wt_amino_acid_code = swap[:3]
        residue_number = int(swap[3: -3])
        var_amino_acid_code = swap[-3:]

        # Convert class to deeprank format (0: benign, 1: pathogenic):
        if variant_class == 0.0:
            variant_class = VariantClass.BENIGN
        else:
            variant_class = VariantClass.PATHOGENIC

        # Iterate over HDF5 table variants, associated with the variant from the parq file (mapped to the ENST code):
        for variant_index, variant_row in variant_table.where(variant_table.ENST == enst_ac).dropna().iterrows():

            # Get associated uniprot and pdb entries:
            uniprot_ac = variant_row["accession"]
            pdb_ac = variant_row["pdb_structure"]
            protein_core_identity = variant_row["protein_core_identity"]

            # filter by core identity
            if protein_core_identity < 70.0:
                continue

            # filter by uniprot accession code
            protein_rows = conservation_table.loc[uniprot_ac]

            # filter by variant sequence residue number
            alignment_positions = protein_rows.where(protein_rows.sequence_residue_number == residue_number)['alignment_position'].dropna()

            # Iterate over the residues that match the variant (usually just one):
            for alignment_position in alignment_positions:

                # filter by pdb accession code
                pdb_rows = conservation_table.loc[pdb_ac]

                # filter by alignment position
                pdb_residue_numbers = pdb_rows.where(pdb_rows.alignment_position == alignment_position)["sequence_residue_number"].dropna()

                for pdb_residue_number in pdb_residue_numbers:
                    pdb_residue_number = int(pdb_residue_number)  # convert from float to int

                    # split up the pdb accession code into an entry and a chain id
                    chain_id = pdb_ac[4]
                    pdb_ac = pdb_ac[:4]

                    _log.debug("encountered {} ({}), mapped to {}-{} residue {}, with core identity {}"
                               .format(variant, variant_class.name, pdb_ac, chain_id, pdb_residue_number, protein_core_identity))

                    pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())
                    if os.path.isfile(pdb_path):

                        pssm_paths = get_pssm_paths(pssm_root, pdb_ac)
                        if len(pssm_paths) == 0:
                            _log.warning("no pssms for: {}".format(pdb_ac))
                            continue

                        # Convert variant to deeprank format:
                        o = PdbVariantSelection(pdb_path, chain_id, pdb_residue_number, AA_codes_3to1[var_amino_acid_code],
                                                pssm_paths, variant_class)
                        objects.add(o)
                    else:
                        _log.warning("no such pdb: {}".format(pdb_path))

    return list(objects)


if __name__ == "__main__":

    variants = get_variant_data(args.variant_path, args.map_path, args.pdb_root, args.pssm_root)

    pathogenic_variants = filter(lambda v: v.variant_class == VariantClass.PATHOGENIC, variants)

    _log.debug("got {} variants, of which {} pathogenic")

    try:
        preprocess(variants, args.out_path)
    except:
        _log.error(traceback.format_exc())
        raise
