from enum import Enum


class VariantClass(Enum):
    BENIGN = 0
    PATHOGENIC = 1


class PdbVariantSelection:
    """Refers to a variant in a pdb file.

    Args:
        pdb_path (str): on disk file path to the pdb file
        chain_id (str): chain within the pdb file, where the variation is
        residue_number (int): the identifying number of the residue within the protein chain
        wild_type_amino_acid (str): one letter code of the wild-type amino acid at this position
        variant_amino_acid (str): one letter code of the amino acid to place at this position
        pssm_paths_by_chain (dict(str, str), optional): the paths of the pssm files per chain id, associated with the pdb file
        variant_class (VariantClass, optional): if known, the expected classification of the variant
        insertion_code (str): insertion code of the residue, default is None
        protein_accession (str): accession code, to identify the protein that this variant was reported in
        protein_residue_number (int): number of the residue in the protein that this variant was reported in
    """

    def __init__(self, pdb_path, chain_id, residue_number,
                 wild_type_amino_acid, variant_amino_acid,
                 pssm_paths_by_chain=None, variant_class=None, insertion_code=None,
                 protein_accession=None, protein_residue_number=None):
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wild_type_amino_acid = wild_type_amino_acid
        self._variant_amino_acid = variant_amino_acid
        self._pssm_paths_by_chain = pssm_paths_by_chain
        self._variant_class = variant_class
        self._protein_accession = protein_accession
        self._protein_residue_number = protein_residue_number

    @property
    def pdb_path(self):
        return self._pdb_path

    def has_pssm(self):
        "are the pssm files included?"
        return self._pssm_paths_by_chain is not None

    def get_pssm_chains(self):
        "returns the chain ids for which pssm files are available"
        if self._pssm_paths_by_chain is not None:
            return self._pssm_paths_by_chain.keys()
        else:
            return set([])

    def get_pssm_path(self, chain_id):
        "returns the pssm path for the given chain id"
        if self._pssm_paths_by_chain is None:
            raise ValueError("pssm paths are not set in this variant selection")

        if chain_id in self._pssm_paths_by_chain:
            return self._pssm_paths_by_chain[chain_id]
        else:
            raise ValueError("{}: no PSSM for chain {}, candidates are {}"
                             .format(self._pdb_path, chain_id, ",".join(self._pssm_paths_by_chain.keys())))

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def residue_number(self):
        return self._residue_number

    @property
    def residue_id(self):
        residue_id = str(self._residue_number)
        if self._insertion_code is not None:
            residue_id += self._insertion_code

        return residue_id

    @property
    def protein_residue_number(self):
        return self._protein_residue_number

    @property
    def insertion_code(self):
        return self._insertion_code

    @property
    def protein_accession(self):
        return self._protein_accession

    @property
    def wild_type_amino_acid(self):
        return self._wild_type_amino_acid

    @property
    def variant_amino_acid(self):
        return self._variant_amino_acid

    def __eq__(self, other):
        return self._pdb_path == other._pdb_path and \
               self._chain_id == other._chain_id and \
               self._residue_number == other._residue_number and \
               self._insertion_code == other._insertion_code and \
               self._wild_type_amino_acid == other._wild_type_amino_acid and \
               self._variant_amino_acid == other._variant_amino_acid

    def __hash__(self):
        s = "pdb={};".format(self._pdb_path) + \
            "chain={};".format(self._chain_id) + \
            "residue_number={}{};".format(self._residue_number, self._insertion_code) + \
            "wild_type_amino_acid={};".format(self._wild_type_amino_acid.name) + \
            "variant_amino_acid={};".format(self._variant_amino_acid.name)

        return hash(s)

    def __repr__(self):
        return "{}:{}:{}:{}->{}".format(self._pdb_path, self._chain_id, self.residue_id, self._wild_type_amino_acid, self._variant_amino_acid)

    @property
    def variant_class(self):
        return self._variant_class
