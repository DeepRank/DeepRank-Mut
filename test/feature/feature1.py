from pdb2sql import pdb2sql
import numpy

from deeprank.operate.pdb import get_pdb_path


def __compute_feature__(environment, featgrp, featgrp_raw, variant):

    pdb_path = get_pdb_path(environment.pdb_root, variant.pdb_ac)

    pdb = pdb2sql(pdb_path)

    try:
        chain_ids = sorted(set(pdb.get("chainID")))
        chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

        data = numpy.array([[x, y, z, numpy.sqrt(numpy.square(x) + numpy.square(y) + numpy.square(z))]
                            for chain_id, x, y, z in pdb.get("chainID,x,y,z")])
        featgrp.create_dataset("feature1", data=data)

    finally:
        pdb._close()
