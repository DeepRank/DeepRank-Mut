from enum import Enum


class VariantClass(Enum):
    BENIGN = 0
    PATHOGENIC = 1


class PdbVariantSelection:
    """Refers to a variant in a pdb file.

    Args:
        pdb_ac (str): pdb accession code
        chain_id (str): chain within the pdb file, where the variation is
        residue_number (int): the identifying number of the residue within the protein chain
        wild_type_amino_acid (str): one letter code of the wild-type amino acid at this position
        variant_amino_acid (str): one letter code of the amino acid to place at this position
        protein_ac (str): protein accession code, used for getting conservations
        protein_residue_number (int): protein residue number, used for getting conservations
        variant_class (VariantClass, optional): if known, the expected classification of the variant
        insertion_code (str): insertion code of the residue, default is None
    """

    def __init__(self, pdb_ac, chain_id, residue_number,
                 wild_type_amino_acid, variant_amino_acid,
                 protein_ac=None, protein_residue_number=None,
                 variant_class=None, insertion_code=None):
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wild_type_amino_acid = wild_type_amino_acid
        self._variant_amino_acid = variant_amino_acid
        self._protein_ac = protein_ac
        self._protein_residue_number = protein_residue_number
        self._variant_class = variant_class

    @property
    def protein_ac(self):
        return self._protein_ac

    @property
    def protein_residue_number(self):
        return self._protein_residue_number

    @property
    def pdb_ac(self):
        return self._pdb_ac

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def residue_number(self):
        return self._residue_number

    @property
    def insertion_code(self):
        return self._insertion_code

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
        residue_id = str(self._residue_number)
        if self._insertion_code is not None:
            residue_id += self._insertion_code

        return "{}:{}:{}:{}->{}".format(self._pdb_path, self._chain_id, residue_id, self._wild_type_amino_acid, self._variant_amino_acid)

    @property
    def variant_class(self):
        return self._variant_class
