#!/usr/bin/env python

import logging
import sys
import os
import gzip
from argparse import ArgumentParser
from threading import Thread
from math import isnan
from glob import glob
import traceback
from time import time

import numpy
import pandas
import csv
from Bio import SeqIO
from mpi4py import MPI
import h5py
from pdb2sql import pdb2sql
import gzip
from pdbecif.mmcif_io import CifFileReader

# Assure that python can find the deeprank files:
deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deeprank_root)

from deeprank.config import logger
from deeprank.generate import DataGenerator
from deeprank.generate.GridTools import GridTools
from deeprank.models.variant import PdbVariantSelection, VariantClass
from deeprank.config.chemicals import AA_codes_3to1, AA_codes_1to3
from deeprank.features.neighbour_profile import WT_FEATURE_NAME, VAR_FEATURE_NAME
from deeprank.operate.conservation import get_conservation_from_bioprodict
from deeprank.operate.hdf5data import load_variant
from deeprank.operate.pdb import get_atoms
from deeprank.models.residue import Residue


arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the (dataset) variant parquet file")
arg_parser.add_argument("map_path", help="the path to the (dataset) mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("pssm_root", help="the path to the pssm root directory, containing files generated with PSSMgen")
arg_parser.add_argument("out_path", help="the path to the output hdf5 file")
arg_parser.add_argument("-A", "--data-augmentation", help="the number of data augmentations", type=int, default=5)
arg_parser.add_argument("-p", "--grid-points", help="the number of points per edge of the 3d grid", type=int, default=20)
arg_parser.add_argument("-S", "--grid-size", help="the length in Angstroms of each edge of the 3d grid", type=float, default=20)



logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.INFO)


mpi_comm = MPI.COMM_WORLD

feature_modules = ["deeprank.features.atomic_contacts",
                   "deeprank.features.accessibility"]
target_modules = ["deeprank.targets.variant_class"]



def annotate_conservation(conservation_dataframe, pdb_dataframe, data_generator):
    """
        Add conservation features from the bioprodict dataframes.

        Args:
            conservation_dataframe (DataFrame): the frame with conservation values from the input hdf5 file
            pdb_dataframe (DataFrame): the frame with pdb mappings from the input hdf5 file
            data_generator (DataGenerator): the deeprank data generator
    """

    logger.info("annotate conservation data for {}".format(data_generator.hdf5))

    with h5py.File(data_generator.hdf5, 'a') as f5:
        for variant_key in f5:
            try:
                logger.info("annotate conservation data for {}".format(variant_key))

                variant_group = f5[variant_key]
                variant = load_variant(variant_group)
                pdb_accession_code = os.path.splitext(os.path.basename(variant.pdb_path))[0][-4:]

                chain_id = variant.chain_id
                chain_conservation_data = get_conservation_from_bioprodict(pdb_dataframe, conservation_dataframe, pdb_accession_code, chain_id)

                db = pdb2sql(variant.pdb_path)
                try:
                    atoms = get_atoms(db)
                    if len(atoms) == 0:
                        logger.error("no atoms for {}".format(variant.pdb_path))

                finally:
                    db._close()

                c_alpha = [atom for atom in atoms if atom.residue.number == variant.residue_number and
                                                     atom.chain_id == variant.chain_id and atom.name == "CA"][0]
                position = c_alpha.position
                residue = c_alpha.residue

                wt = variant.wild_type_amino_acid
                var = variant.variant_amino_acid

                if residue not in chain_conservation_data:
                    logger.error("{} is not in the conservation data, candidates are: {}"
                                 .format(residue, '\n'.join([str(r) for r in chain_conservation_data.keys()])))
                    del f5[variant_key]

                    continue

                wt_data = numpy.array([list(position) + [chain_conservation_data[residue][wt]]])
                var_data = numpy.array([list(position) + [chain_conservation_data[residue][var]]])

                feature_group = variant_group.require_group("features")

                feature_group.create_dataset(WT_FEATURE_NAME, data=wt_data)
                feature_group.create_dataset(VAR_FEATURE_NAME, data=var_data)
            except:
                logger.error(traceback.format_exc())
                continue


def preprocess(conservation_dataframe, pdb_dataframe, variants, hdf5_path, data_augmentation, grid_info):
    """ Generate the preprocessed data as an hdf5 file

        Args:
            variants (list of PdbVariantSelection objects): the variant data
            hdf5_path (str): the output HDF5 path
            data_augmentation (int): the number of data augmentations per variant
            grid_info (dict): the settings for the grid
    """

    data_generator = DataGenerator(variants,
                                   compute_features=feature_modules,
                                   compute_targets=target_modules,
                                   hdf5=hdf5_path, mpi_comm=mpi_comm,
                                   data_augmentation=data_augmentation)
    data_generator.create_database()
    annotate_conservation(conservation_dataframe, pdb_dataframe, data_generator)

    data_generator.map_features(grid_info)


# in conservation table:
# needed:  pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
# present:  amino_acid  sub_sequencecount  sub_consv_A  sub_consv_B  sub_consv_C  sub_consv_D  ...  sub_consv_V  sub_consv_W  sub_consv_X  sub_consv_Y  sub_consv_Z  sub_consv_gap


def get_variant_data(class_table, variant_table, conservation_table, pdb_table, pdb_root, pssm_root):
    """ Extract data from the dataset and convert to variant objects.

        Args:
            class_table (DataFrame): the table of variant classes from the parq file
            variant_table (DataFrame): the table of variant mappings from the input hdf5 file
            conservation_table (DataFrame): the table of conservations from the input hdf5 file
            pdb_table (DataFrame): the table of pdb mappings from the inpur hdf5 file
            pdb_root (str): path to the directory where the pdb files are located as: pdb????.ent
            pssm_root (str): path to the directory where the PSSMgen output files are located

        Returns (list of PdbVariantSelection objects): the variants in the dataset
        Raises (ValueError): if data is inconsistent
    """

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
        elif variant_class == 1.0:
            variant_class = VariantClass.PATHOGENIC
        else:
            raise ValueError("Unknown class: {}".format(variant_class))

        # Iterate over HDF5 table variants, associated with the variant from the parq file (mapped to the ENST code):
        for variant_index, variant_row in variant_table.where(variant_table.ENST == enst_ac).dropna().iterrows():

            # Get associated uniprot and pdb entries:
            uniprot_ac = variant_row["accession"]
            pdb_ac = variant_row["pdb_structure"]
            protein_core_identity = variant_row["protein_core_identity"]

            if len(pdb_ac) != 5:
                raise ValueError("No valid pdb accession in variant row: {}".format(variant_row))

#            # filter by core identity
#            if protein_core_identity < 70.0:
#                continue

            logger.info("got variant {}, associated with {} and {}".format(variant, uniprot_ac, pdb_ac))

            # filter by uniprot accession code
            protein_rows = conservation_table.loc[uniprot_ac]

            # filter by variant sequence residue number
            alignments = protein_rows.where(protein_rows.sequence_residue_number == residue_number).dropna()

            # Iterate over the residues that match the variant (usually just one):
            for alignment_index, alignment_row in alignments.iterrows():

                alignment_position = alignment_row['alignment_position']

                # filter by pdb accession code
                pdb_rows = pdb_table.loc[pdb_ac]

                # filter by alignment position
                pdb_residues = pdb_rows.where(pdb_rows.alignment_position == alignment_position).dropna()

                for pdb_residue_index, pdb_residue in pdb_residues.iterrows():
                    pdb_number = int(pdb_residue['pdbnumber'])  # convert from float to int

                    # split up the pdb accession code into an entry and a chain id
                    chain_id = pdb_ac[4]
                    pdb_ac = pdb_ac[:4]

                    logger.debug("encountered {} ({}), mapped to {}-{} residue {}, with core identity {}"
                                 .format(variant, variant_class.name, pdb_ac, chain_id, pdb_number, protein_core_identity))

                    pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())

                    pssm_paths = {}  # we take this from the bio-prodict files

                    # Convert variant to deeprank format:
                    o = PdbVariantSelection(pdb_path, chain_id, pdb_number,
                                            AA_codes_3to1[wt_amino_acid_code], AA_codes_3to1[var_amino_acid_code],
                                            pssm_paths, variant_class)
                    objects.add(o)

    return list(objects)


def get_subset(variants):
    """ Take a subset of the input list of variants so that the ratio benign/pathogenic is 50 / 50

        Args:
            variants (list of PdbVariantSelection objects): the input variants

        Returns (list of PdbVariantSelection objects): the subset of variants, taken from the input
    """

    benign = []
    pathogenic = []
    for variant in variants:
        if variant.variant_class == VariantClass.PATHOGENIC:
            pathogenic.append(variant)
        elif variant.variant_class == VariantClass.BENIGN:
            benign.append(variant)

    logger.info("variants: got {} benign and {} pathogenic".format(len(benign), len(pathogenic)))

    count = min(len(benign), len(pathogenic))

    numpy.random.seed(int(time()))

    numpy.random.shuffle(benign)
    numpy.random.shuffle(pathogenic)

    variants = benign[:count] + pathogenic[:count]
    numpy.random.shuffle(variants)

    logger.info("variants: taking a subset of {}".format(len(variants)))

    return variants


if __name__ == "__main__":
    args = arg_parser.parse_args()

    resolution = args.grid_size / args.grid_points

    grid_info = {
       'number_of_points': [args.grid_points, args.grid_points, args.grid_points],
       'resolution': [resolution, resolution, resolution],
       'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
    }

    class_table = pandas.read_parquet(args.variant_path)
    variant_table = pandas.read_hdf(args.map_path, "variants")
    conservation_table = pandas.read_hdf(args.map_path, "conservation")
    pdb_table = pandas.read_hdf(args.map_path, "pdbs")

    variants = get_variant_data(class_table, variant_table, conservation_table, pdb_table, args.pdb_root, args.pssm_root)

    variants = get_subset(variants)

    try:
        preprocess(conservation_table, pdb_table, variants, args.out_path, args.data_augmentation, grid_info)
    except:
        logger.error(traceback.format_exc())
        raise
