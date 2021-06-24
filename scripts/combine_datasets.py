import sys
from argparse import ArgumentParser

import h5py


arg_parser = ArgumentParser(description="combine hdf5 files into one")
arg_parser.add_argument("-i", "--input-file", action="append", help="hdf5 input files")
arg_parser.add_argument("output_file", help="hdf5 output file")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    with h5py.File(args.output_file, 'w') as output:
        for path in args.input_file:
            with h5py.File(path, 'r') as input_:
                for group_name in input_.keys():
                    g = output.create_group(group_name)
                    input_.copy(group_name, g)
