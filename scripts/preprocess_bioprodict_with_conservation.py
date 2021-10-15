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
from deeprank.models.variant import PdbVariantSelection, VariantClass
from deeprank.config.chemicals import AA_codes_3to1
from deeprank.domain.amino_acid import amino_acids
from deeprank.operate.hdf5data import load_variant
from deeprank.operate.pdb import get_atoms
from deeprank.features.neighbour_profile import WT_FEATURE_NAME, VAR_FEATURE_NAME

arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the (dataset) variant parquet file")
arg_parser.add_argument("map_path", help="the path to the (dataset) mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("out_path", help="the path to the output hdf5 file")
arg_parser.add_argument("-A", "--data-augmentation", help="the number of data augmentations", type=int, default=5)
arg_parser.add_argument("-p", "--grid-points", help="the number of points per edge of the 3d grid", type=int, default=20)
arg_parser.add_argument("-S", "--grid-size", help="the length in Angstroms of each edge of the 3d grid", type=float, default=20)



logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.INFO)
_log = logging.getLogger(__name__)


mpi_comm = MPI.COMM_WORLD

feature_modules = ["deeprank.features.atomic_contacts",
                   "deeprank.features.accessibility"]
target_modules = ["deeprank.targets.variant_class"]


def preprocess(variants, hdf5_path, data_augmentation, grid_info, conservations):
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

    add_conservation(conservations, data_generator.hdf5)

    data_generator.map_features(grid_info)


def add_conservation(conservations, output_hdf5):

    with h5py.File(output_hdf5, 'a') as f5:
        for variant_group_name in f5.keys():
            variant_group = f5[variant_group_name]
            variant = load_variant(variant_group)
            if variant in conservations:
                wt = variant.wild_type_amino_acid
                var = variant.variant_amino_acid

                pdb = pdb2sql(variant.pdb_path)
                try:
                    atoms = get_atoms(pdb)
                    if len(atoms) == 0:
                        logger.error("no atoms for {}".format(variant.pdb_path))
                finally:
                    pdb._close()

                c_alpha = [atom for atom in atoms if atom.residue.number == variant.residue_number and 
                                                     atom.chain_id == variant.chain_id and atom.name == "CA"][0]
                position = c_alpha.position

                wt_data = numpy.array([list(position) + [conservations[variant][wt]]])
                var_data = numpy.array([list(position) + [conservations[variant][var]]])

                feature_group = variant_group.require_group("features")

                _log.debug("add conservation to {}".format(variant_group_name))

                feature_group.create_dataset(WT_FEATURE_NAME, data=wt_data)
                feature_group.create_dataset(VAR_FEATURE_NAME, data=var_data)
            else:
                _log.warning("deleting {} ({}) because there's no conservation".format(variant_group_name, variant))
                del f5[variant_group_name]


# in conservation table:
# needed:  pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
# present:  amino_acid  sub_sequencecount  sub_consv_A  sub_consv_B  sub_consv_C  sub_consv_D  ...  sub_consv_V  sub_consv_W  sub_consv_X  sub_consv_Y  sub_consv_Z  sub_consv_gap


_VARIANT_NAME_COLUMN = "variant"
_VARIANT_CLASS_COLUMN = "class"
_PDB_AC_COLUMN = "pdb_structure"
_PDB_NUMBER_COLUMN = "pdbnumber"

def get_variant_data(parq_path, hdf5_path, pdb_root):
    """ Extract data from the dataset and convert to variant objects.

        Args:
            parq_path (str): path to the bioprodict parq file, containing the variants
            hdf5_path (str): path to the bioprodict hdf5 file, mapping the variants to pdb entries
            pdb_root (str): path to the directory where the pdb files are located as: pdb????.ent

        Returns (list of PdbVariantSelection objects): the variants in the dataset
        Raises (ValueError): if data is inconsistent
    """

    amino_acids_by_code = {amino_acid.code: amino_acid for amino_acid in amino_acids}

    class_table = pandas.read_parquet(parq_path)
    mappings_table = pandas.read_hdf(hdf5_path, "mappings")

    objects = set([])

    # Get all variants in the parq file:
    for column_name in [_VARIANT_NAME_COLUMN, _VARIANT_CLASS_COLUMN]:
        if column_name not in class_table.columns:
            raise ValueError("{} has no column named {}".format(parq_path, column_name))

    variant_classes = {}
    for row_index, row in class_table.iterrows():
        variant_name = row[_VARIANT_NAME_COLUMN]
        variant_class = row[_VARIANT_CLASS_COLUMN]

        # Convert class to deeprank format (0: benign, 1: pathogenic):
        if variant_class == 0.0:
            variant_class = VariantClass.BENIGN

        elif variant_class == 1.0:
            variant_class = VariantClass.PATHOGENIC
        else:
            raise ValueError("Unknown class: {}".format(variant_class))

        variant_classes[variant_name] = variant_class

    # Get all mappings to pdb and use them to create variant objects:
    for column_name in [_VARIANT_NAME_COLUMN, _PDB_AC_COLUMN, _PDB_NUMBER_COLUMN]:
        if column_name not in mappings_table.columns:
            raise ValueError("{}/{} has no column named {}".format(hdf5_path, "mappings", column_name))

    objects = set([])

    protein_variants = {}

    for _, variant_row in mappings_table.iterrows():

        variant_name = variant_row["variant"]
        if variant_name not in variant_classes:
            _log.warning("no such variant: {}".format(variant_name))
            continue

        variant_class = variant_classes[variant_name]

        enst_ac = variant_name[:15]
        swap = variant_name[15:]
        wt_amino_acid_code = swap[:3]
        residue_number = int(swap[3: -3])
        var_amino_acid_code = swap[-3:]

        pdb_ac = variant_row["pdb_structure"]

        pdb_number_string = variant_row["pdbnumber"]
        if pdb_number_string[-1].isalpha():

            pdb_number = int(pdb_number_string[:-1])
            insertion_code = pdb_number_string[-1]
        else:
            pdb_number = int(pdb_number_string)
            insertion_code = None

        chain_id = pdb_ac[4]
        pdb_ac = pdb_ac[:4]

        pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())
        if not os.path.isfile(pdb_path):
            _log.warning("no such pdb: {}".format(pdb_path))
            continue

        _log.info("add variant on {} {} {} {}->{} = {}"
                  .format(pdb_path, chain_id, pdb_number,
                          wt_amino_acid_code, var_amino_acid_code,
                          variant_class))

        protein_ac = variant_row['protein_accession']

        o = PdbVariantSelection(pdb_path, chain_id, pdb_number,
                                amino_acids_by_code[wt_amino_acid_code],
                                amino_acids_by_code[var_amino_acid_code],
                                {}, variant_class, insertion_code)
        objects.add(o)

        protein_variants[(protein_ac, residue_number)] = o

    conservations = {}
    protein_ac = None
    prev_ac = None
    residue_number = None
    conservation_table = pandas.read_hdf(hdf5_path, "conservation")
    for _, conservation_row in conservation_table.iterrows():
        protein_ac = conservation_row['accession']
        if protein_ac != prev_ac:
            residue_number = 1
        else:
            residue_number += 1

        key = (protein_ac, residue_number)
        if key in protein_variants:
            variant = protein_variants[key]

            _log.debug("conservation for {}".format(variant))
            conservations[variant] = {amino_acid: conservation_row["sub_consv_{}".format(amino_acid.letter)]
                                      for amino_acid in amino_acids}

        prev_ac = protein_ac

    return objects, conservations


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

    variants, conservations = get_variant_data(args.variant_path, args.map_path, args.pdb_root)

    variants = get_subset(variants)

    try:
        preprocess(variants, args.out_path, args.data_augmentation, grid_info, conservations)
    except:
        logger.error(traceback.format_exc())
        raise
