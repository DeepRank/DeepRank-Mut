

class Environment:
    def __init__(self,
                 pdb_root=None,
                 pssm_root=None,
                 dssp_root=None,
                 conservation_root=None,
                 dbnsfp_path=None,
                 gnomad_path=None,
                 device=None,
                 zero_missing_pssm=False):

        self.pdb_root = pdb_root
        self.pssm_root = pssm_root
        self.dssp_root = dssp_root
        self.conservation_root = conservation_root
        self.dbnsfp_path = dbnsfp_path
        self.gnomad_path = gnomad_path
        self.device = device
        self.zero_missing_pssm = zero_missing_pssm
