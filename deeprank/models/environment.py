

class Environment:
    def __init__(self,
                 pdb_root=None,
                 pssm_root=None,
                 conservation_root=None,
                 device=None):

        self.pdb_root = pdb_root
        self.pssm_root = pssm_root
        self.conservation_root = conservation_root
        self.device = device
