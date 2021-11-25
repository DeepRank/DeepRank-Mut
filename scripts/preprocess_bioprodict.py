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
import torch.cuda

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
from deeprank.operate.pdb import is_xray
from deeprank.models.environment import Environment


arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the (dataset) variant parquet file")
arg_parser.add_argument("map_path", help="the path to the (dataset) mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("--pssm-root", help="the path to the pssm root directory, containing files generated with PSSMgen")
arg_parser.add_argument("--conservation-root", help="the path to the conservations root directory, containing conservation files per protein")
arg_parser.add_argument("out_path", help="the path to the output hdf5 file")
arg_parser.add_argument("-A", "--data-augmentation", help="the number of data augmentations", type=int, default=5)
arg_parser.add_argument("-p", "--grid-points", help="the number of points per edge of the 3d grid", type=int, default=20)
arg_parser.add_argument("-S", "--grid-size", help="the length in Angstroms of each edge of the 3d grid", type=float, default=20)



logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.INFO)
_log = logging.getLogger(__name__)


mpi_comm = MPI.COMM_WORLD


def preprocess(environment, variants, hdf5_path, data_augmentation, grid_info,
               feature_modules, target_modules):
    """ Generate the preprocessed data as an hdf5 file

        Args:
            environment (Environment): the environment settings
            variants (list of PdbVariantSelection objects): the variant data
            hdf5_path (str): the output HDF5 path
            data_augmentation (int): the number of data augmentations per variant
            grid_info (dict): the settings for the grid
            feature_modules (list of strings): names of the feature modules to include
            target_modules (list of strings): names of the target modules to include
    """

    data_generator = DataGenerator(environment, variants,
                                   compute_features=feature_modules,
                                   compute_targets=target_modules,
                                   hdf5=hdf5_path, mpi_comm=mpi_comm,
                                   data_augmentation=data_augmentation)
    data_generator.create_database()
    data_generator.map_features(grid_info)


def pdb_meets_criteria(pdb_root, pdb_ac):
    "a set of criteria that every pdb entry should meet"

    pdb_paths = glob(os.path.join(pdb_root, "*{}.*".format(pdb_ac.lower()))) + \
                glob(os.path.join(pdb_root, "*{}.*".format(pdb_ac.upper())))

    if len(pdb_paths) == 0:
        _log.warning("no pdb found for {}".format(pdb_ac))
        return False

    elif len(pdb_paths) > 1:
        _log.warning("more than one pdb file for {}: {}".format(pdb_ac, pdb_paths))
        return False

    with open(pdb_paths[0], 'rt') as f:
        if not is_xray(f):
            _log.warning("not an x-ray pdb: {}".format(pdb_paths[0]))
            return False

    return True


def has_pssm(pssm_root, pdb_ac):
    paths = glob(os.path.join(pssm_root, "{}/pssm/{}.?.pdb.pssm".format(pdb_ac.lower(), pdb_ac.lower())))

    if len(paths) == 0:
        _log.warning("no pssms for {}".format(pdb_ac))
        return False

    return True


def has_conservation(conservation_root, protein_ac):
    paths = [os.path.join(conservation_root, "{}.parq".format(protein_ac)),
             os.path.join(conservation_root, "{}.PARQ".format(protein_ac))]

    if not any([os.path.isfile(path) for path in paths]):
        _log.warning("no conservation for {}".format(protein_ac))
        return False

    return True


# in conservation table:
# needed:  pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
# present:  amino_acid  sub_sequencecount  sub_consv_A  sub_consv_B  sub_consv_C  sub_consv_D  ...  sub_consv_V  sub_consv_W  sub_consv_X  sub_consv_Y  sub_consv_Z  sub_consv_gap


_VARIANT_NAME_COLUMN = "variant"
_VARIANT_CLASS_COLUMN = "class"
_PDB_AC_COLUMN = "pdb_structure"
_PDB_NUMBER_COLUMN = "pdbnumber"


def get_variant_data(parq_path):
    """ extracts variant names and truth values(classes from a parquet file)

        Args:
            parq_path(str): path to the parquet file in propert format
        Returns (list((str, VariantClass)): the variant names and classes
    """

    variant_data = []

    _log.debug("reading {}".format(parq_path))

    class_table = pandas.read_parquet(parq_path)

    for _, row in class_table.iterrows():
        variant_name = row[_VARIANT_NAME_COLUMN]
        variant_class = row[_VARIANT_CLASS_COLUMN]

        # Convert class to deeprank format (0: benign, 1: pathogenic):
        if variant_class == 0.0:
            variant_class = VariantClass.BENIGN

        elif variant_class == 1.0:
            variant_class = VariantClass.PATHOGENIC
        else:
            raise ValueError("Unknown class: {}".format(variant_class))

        _log.debug("add variant {} {}".format(variant_name, variant_class))

        variant_data.append((variant_name, variant_class))

    return variant_data


def get_mappings(hdf5_path, pdb_root, pssm_root, conservation_root, variant_data):
    """ read the hdf5 file to map variant data to pdb and pssm data

        Args:
            hdf5_path(str): path to an hdf5 file, containing a table named "mappings"
            pdb_root(str): path to the directory where pdbs are stored
            pssm_root(str or None): path to the directory where pssms are stored
            conservation_root(str or None): path to the directory where conservation parquet tables are stored
            variant_data (list((str, VariantClass)): the variant names and classes

        Returns (set(PdbVariantSelection)): the variant objects that deeprank will use
    """

    amino_acids_by_code = {amino_acid.code: amino_acid for amino_acid in amino_acids}

    _log.debug("reading {} mappings table".format(hdf5_path))

    mappings_table = pandas.read_hdf(hdf5_path, "mappings")

    variants = set([])

    for variant_name, variant_class in variant_data:
        variant_section = mappings_table.loc[mappings_table.variant == variant_name].dropna()

        enst_ac = variant_name[:15]
        swap = variant_name[15:]
        wt_amino_acid_code = swap[:3]
        residue_number = int(swap[3: -3])
        var_amino_acid_code = swap[-3:]

        variants_section= variant_section.iloc[:20] 

        for _, row in variants_section.iterrows():  # each row maps the variant to one pdb entry

            pdb_ac = row["pdb_structure"]

            pdb_number_string = row["pdbnumber"]
            if pdb_number_string[-1].isalpha():

                pdb_number = int(pdb_number_string[:-1])
                insertion_code = pdb_number_string[-1]
            else:
                pdb_number = int(pdb_number_string)
                insertion_code = None

            chain_id = pdb_ac[4]
            pdb_ac = pdb_ac[:4]

            if not pdb_meets_criteria(pdb_root, pdb_ac):
                continue

            if pssm_root is not None and not has_pssm(pssm_root, pdb_ac):
                continue

            protein_ac = row["protein_accession"]
            if conservation_root is not None and not has_conservation(conservation_root, protein_ac):
                continue

            variant = PdbVariantSelection(pdb_ac, chain_id, pdb_number,
                                          amino_acids_by_code[wt_amino_acid_code],
                                          amino_acids_by_code[var_amino_acid_code],
                                          protein_ac, residue_number,
                                          variant_class, insertion_code)

            _log.debug("adding variant {}".format(variant))

            variants.add(variant)

    return variants


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

    variants_data = get_variant_data(args.variant_path)

    variants = get_mappings(args.map_path, args.pdb_root, args.pssm_root, args.conservation_root, variants_data)

    variants = get_subset(variants)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    environment = Environment(args.pdb_root, args.pssm_root, args.conservation_root, device)

    feature_modules = ["deeprank.features.atomic_contacts",
                       "deeprank.features.accessibility"]
    target_modules = ["deeprank.targets.variant_class"]

    if args.conservation_root is not None:
        feature_modules.append("deeprank.features.variant_conservation")

    if args.pssm_root is not None:
        feature_modules.append("deeprank.features.neighbour_profile")

    try:
        preprocess(environment, variants, args.out_path, args.data_augmentation, grid_info,
                   feature_modules, target_modules)
    except:
        logger.error(traceback.format_exc())
        raise
