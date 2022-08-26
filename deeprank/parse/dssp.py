from typing import Dict

from deeprank.models.residue import Residue
from deeprank.models.secondary_structure import SecondaryStructure
from deeprank.domain.amino_acid import amino_acids, cysteine


def _is_dssp_header(line: str) -> bool:
    #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA

    if line is None:
        return False

    return line.strip().startswith("#")


def parse_dssp(path: str) -> Dict[Residue, SecondaryStructure]:
    "parse one dssp file and return the secondary structure by residue"

    amino_acids_by_letter = {amino_acid.letter: amino_acid for amino_acid in amino_acids}

    dssp = {}

    with open(path, 'rt') as f:

        header_found = False
        for line in f:

            if _is_dssp_header(line):
                header_found = True

            elif header_found and len(line.strip()) > 0:
                residue_id = line[5: 11]
                if len(residue_id.strip()) == 0:
                    continue  # skip chain breaks

                if residue_id[-1].isalpha():
                    insertion_code = residue_id[-1]
                    residue_number = int(residue_id[:-1])
                else:
                    insertion_code = None
                    residue_number = int(residue_id)

                chain_id = line[11]

                amino_acid_letter = line[13]
                if amino_acid_letter.islower():
                    amino_acid = cysteine
                else:
                    amino_acid = amino_acids_by_letter[amino_acid_letter]

                residue = Residue(residue_number, amino_acid.name, chain_id, insertion_code)

                secondary_structure_string = line[16]
                if secondary_structure_string == "H":
                    secondary_structure = SecondaryStructure.HELIX

                elif secondary_structure_string == "E":
                    secondary_structure = SecondaryStructure.STRAND
                else:
                    secondary_structure = SecondaryStructure.OTHER

                dssp[residue] = secondary_structure

    return dssp
