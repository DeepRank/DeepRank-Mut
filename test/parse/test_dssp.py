from deeprank.parse.dssp import parse_dssp
from deeprank.models.secondary_structure import SecondaryStructure
from deeprank.models.residue import Residue


def test_parse_1crn():

    dssp_path = "test/data/dssp/1crn.dssp"

    dssp = parse_dssp(dssp_path)

    residue_2 = Residue(2, "Threonine", "A")
    assert dssp[residue_2] == SecondaryStructure.STRAND, f"{residue_2} is {dssp[residue_2]} in {dssp_path}"

    count_helix = 0
    count_strand = 0
    for residue, secondary_structure in dssp.items():
        if secondary_structure == SecondaryStructure.HELIX:
            count_helix += 1

        elif secondary_structure == SecondaryStructure.STRAND:
            count_strand += 1

    assert count_helix >= 10, f"only {count_helix} helix residues in {dssp_path}"
    assert count_strand >= 4, f"only {count_strand} strand residues in {dssp_path}"

