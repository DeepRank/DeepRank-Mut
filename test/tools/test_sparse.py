import tempfile
import os

import numpy
import numpy.testing
import h5py

from deeprank.tools.sparse import FLANgrid


def test_preserved():

    beta = 1E-2

    data = numpy.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])

    spg = FLANgrid()
    spg.from_dense(data, beta=beta)

    hdf5_file, hdf5_path = tempfile.mkstemp()
    os.close(hdf5_file)
    with h5py.File(hdf5_path, 'w') as f5:
        f5.create_dataset("index", data=spg.index, compression='gzip', compression_opts=9)
        f5.create_dataset("value", data=spg.value, compression='gzip', compression_opts=9)

    with h5py.File(hdf5_path, 'r') as f5:
        spg_retrieve = FLANgrid(True, f5['index'][:], f5['value'][:], data.shape)

    retrieved_data = spg_retrieve.to_dense()

    assert numpy.all(numpy.abs(data - retrieved_data) < beta), "{} is not the same as {}".format(data, retrieved_data)


