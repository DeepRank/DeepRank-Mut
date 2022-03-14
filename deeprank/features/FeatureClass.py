import numpy as np

class FeatureClass(object):

    def __init__(self, feature_type):
        """Master class from which all the other feature classes should be derived.

        Arguments
            feature_type(str): 'Atomic' or 'Residue'

        Note:
            Each subclass must compute:

            - self.feature_data: dictionary of features in human readable format, e.g.
                - for atomic features:
                    - {'coulomb': data_dict_clb, 'vdwaals': data_dict_vdw}
                    - data_dict_clb = {atom_info: [values]}
                    - atom_info = (chainID, resSeq, resName, name)
                - for residue features:
                    - {'PSSM_ALA': data_dict_pssmALA, ...}
                    - data_dict_pssmALA = {residue_info: [values]}
                    - residue_info = (chainID, resSeq, resName, name)
            - self.feature_data_xyz: dictionary of features in xyz-val format, e.g.
                - {'coulomb': data_dict_clb, 'vdwaals': data_dict_vdw}
                - data_dict_clb = {xyz_info: [values]}
                - xyz_info = (chainNum, x, y, z)
        """

        self.type = feature_type
        self.feature_data = {}
        self.feature_data_xyz = {}

    def export_dataxyz_hdf5(self, featgrp):
        """Export the data in xyz-val format in an HDF5 file group.

        Arguments:
            featgrp {[hdf5_group]} -- The hdf5 group of the feature
        """

        # loop through the datadict and name
        for name, data in self.feature_data_xyz.items():

            # create the data set
            ds = np.array([list(key) + value for key, value in data.items()])

            # create the dataset
            if name in featgrp:
                old = featgrp[name]
                old[...] = ds
            else:
                featgrp.create_dataset(name, data=ds, chunks=True)

    @staticmethod
    def get_residue_center(sql, centers=['CB','CA','mean'], res=None):
        """Computes the center of each residue by trying different options

        Arguments:
            sql {pdb2sql} -- The pdb2sql instance

        Keyword Arguments:
            centers {list} -- list of strings (default: {['CB','CA','mean']})
            res {list} -- list of residue to be considered ([[chainID, resSeq, resName]])

        Raises:
            ValueError: [description]

        Returns:
            [type] -- list(res), list(xyz)
        """

        # get all residues if None were provided
        # [chainID, resName, resSeq]
        if res is None:
            res = [tuple(x) for x in sql.get('chainID,resSeq,resName')]
            res = sorted(set(res), key=res.index)


        # make sure that we have a list of res
        # even if ony 1 res was provided
        # res=[chainID, resSeq, resName] -> res=[[chainID, resSeq, resName]]
        elif not isinstance(res[0],list):
            res = [res]

        # make sure that we have a list of possible centers
        if not isinstance(centers,list):
            centers = list(centers)

        xyz = []

        for r in res:

            for ctr in centers:

                if ctr in ['CB','CA']:

                    xyz_res = sql.get('x,y,z',
                                      chainID=r[0],
                                      resSeq=r[1],
                                      resName=r[2],
                                      name=ctr)

                elif ctr == 'mean':
                    xyz_res = [np.mean(sql.get('x,y,z',
                                       chainID=r[0],
                                       resSeq=r[1],
                                       resName=r[2]),axis=0).tolist()]

                else:
                    raise ValueError('Center %s not recognized' %c)

                if len(xyz_res) == 0:
                    continue

                elif len(xyz_res) == 1:
                    xyz.append(xyz_res[0])
                    break

                else:
                    raise ValueError('Residue center not found')

        if len(xyz) == 0:
            raise ValueError('Center not found')

        return res, xyz
