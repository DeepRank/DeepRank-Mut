

class Environment:
    def __init__(self,
                 pdb_root=None,
                 pssm_root=None,
                 conservation_root=None,
                 dbnsfp_path=None,
                 device=None):

        self.pdb_root = pdb_root
        self.pssm_root = pssm_root
        self.conservation_root = conservation_root
        self.dbnsfp_path = dbnsfp_path
        self.device = device
