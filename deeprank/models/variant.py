from enum import Enum
from typing import Optional

from deeprank.models.amino_acid import AminoAcid


class VariantClass(Enum):
    UNKNOWN = -1
    BENIGN = 0
    PATHOGENIC = 1


class PdbVariantSelection:
    """Refers to a variant in a pdb file.

    Args:
        pdb_ac: pdb accession code
        chain_id: chain within the pdb file, where the variation is
        residue_number: the identifying number of the residue within the protein chain
        wildtype_amino_acid: one letter code of the wild-type amino acid at this position
        variant_amino_acid: one letter code of the amino acid to place at this position
        enst_accession: ensemble transcript accession code, used for database search
        protein_accession: protein accession code, used for getting conservations
        protein_residue_number: protein residue number, used for getting conservations
        variant_class: if known, the expected classification of the variant
        insertion_code: insertion code of the residue, default is None
    """

    def __init__(self, pdb_ac: str, chain_id: str, residue_number: int,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 enst_accession: Optional[str] = None,
                 protein_accession: Optional[str] = None, protein_residue_number: Optional[int] = None,
                 variant_class: Optional[VariantClass] = None, insertion_code: Optional[str] = None):

        self._pdb_ac = pdb_ac
        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid
        self._enst_accession = enst_accession
        self._protein_accession = protein_accession
        self._protein_residue_number = protein_residue_number
        self._variant_class = variant_class

    @property
    def protein_accession(self):
        return self._protein_accession

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
    def residue_id(self):
        residue_id = str(self._residue_number)
        if self._insertion_code is not None:
            residue_id += self._insertion_code

        return residue_id

    @property
    def insertion_code(self):
        return self._insertion_code

    @property
    def enst_accession(self):
        return self._enst_accession

    @property
    def wild_type_amino_acid(self):
        return self._wildtype_amino_acid

    @property
    def wildtype_amino_acid(self):
        return self._wildtype_amino_acid

    @property
    def variant_amino_acid(self):
        return self._variant_amino_acid

    def __eq__(self, other):
        return self._pdb_ac == other._pdb_ac and \
               self._chain_id == other._chain_id and \
               self._residue_number == other._residue_number and \
               self._insertion_code == other._insertion_code and \
               self._wildtype_amino_acid == other._wildtype_amino_acid and \
               self._variant_amino_acid == other._variant_amino_acid

    def __hash__(self):
        s = "pdb={};".format(self._pdb_ac) + \
            "chain={};".format(self._chain_id) + \
            "residue_number={}{};".format(self._residue_number, self._insertion_code) + \
            "wildtype_amino_acid={};".format(self._wildtype_amino_acid.name) + \
            "variant_amino_acid={};".format(self._variant_amino_acid.name)

        return hash(s)

    def __repr__(self):
        return "{}:{}:{}:{}->{}".format(self._pdb_ac, self._chain_id, self.residue_id, self._wildtype_amino_acid, self._variant_amino_acid)

    @property
    def variant_class(self):
        return self._variant_class
