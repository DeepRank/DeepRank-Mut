import sys
from argparse import ArgumentParser
import logging

import h5py


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="combine hdf5 files into one")
arg_parser.add_argument("-i", "--input-file", action="append", help="hdf5 input files")
arg_parser.add_argument("output_file", help="hdf5 output file")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    with h5py.File(args.output_file, 'w') as output:
        for path in args.input_file:
            with h5py.File(path, 'r') as input_:
                for group_name, group in input_.items():

                    _log.debug("copy group {} from {} to {}".format(group_name, path, args.output_file))

                    input_.copy(group_name, output, group_name)
