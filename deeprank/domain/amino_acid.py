from deeprank.models.amino_acid import AminoAcid
from deeprank.models.polarity import Polarity


alanine = AminoAcid("Alanine", "ALA", "A",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=1,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

arginine = AminoAcid("Arginine", "ARG", "R",
    charge=-1.65,
    polarity=Polarity.POSITIVE_CHARGE,
    size=7,
    count_hydrogen_bond_donors=5,
    count_hydrogen_bond_acceptors=0
)

asparagine = AminoAcid("Asparagine", "ASN", "N",
    charge=-1.22,
    polarity=Polarity.POLAR,
    size=4,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2
)

aspartate = AminoAcid("Aspartate", "ASP", "D",
    charge=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4
)

protonated_aspartate = AminoAcid("Protonated Aspartate", "ASH", "D",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=4,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=3
)

cysteine = AminoAcid("Cysteine", "CYS", "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

cysteine_metal = AminoAcid("Cysteine Metal Ligand", "CYM", "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

cysteine_iron = AminoAcid("Cysteine Iron Ligand", "CFE", "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

cysteine_zinc = AminoAcid("Cysteine Zinc Ligand", "CYF", "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

cysteine_phosphate = AminoAcid("Cysteine Phosphate", "CSP", "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)
glutamate = AminoAcid("Glutamate", "GLU", "E",
    charge=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=5,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4,
)
protonated_glutamate = AminoAcid("Protonated Glutamate", "GLH", "E",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=4,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=3
)

glutamine = AminoAcid("Glutamine", "GLN", "Q",
    charge=-1.22,
    polarity=Polarity.POLAR,
    size=5,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2,
)

glycine = AminoAcid("Glycine", "GLY", "G",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=0,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

histidine = AminoAcid("Histidine", "HIS", "H",
    charge=-0.29,
    polarity=Polarity.POLAR,
    size=6,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2
)

histidine_phosphate = AminoAcid("Histidine Phosphate", "NEP", "H",
    charge=-0.29,
    polarity=Polarity.POLAR,
    size=6,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2
)

isoleucine = AminoAcid("Isoleucine", "ILE", "I",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
)

leucine = AminoAcid("Leucine", "LEU", "L",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

lysine = AminoAcid("Lysine", "LYS", "K",
    charge=-0.36,
    polarity=Polarity.POSITIVE_CHARGE,
    size=5,
    count_hydrogen_bond_donors=3,
    count_hydrogen_bond_acceptors=0
)

methionine = AminoAcid("Methionine", "MET", "M",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

phenylalanine = AminoAcid("Phenylalanine", "PHE", "F",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=7,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

proline = AminoAcid("Proline", "PRO", "P",
    charge=0.0,
    polarity=Polarity.APOLAR,
    size=3,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

serine = AminoAcid("Serine", "SER", "S",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2
)

threonine = AminoAcid("Threonine", "THR", "T",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=3,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2
)

tryptophan = AminoAcid("Tryptophan", "TRP", "W",
    charge=-0.79,
    polarity=Polarity.POLAR,
    size=10,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=0
)

tyrosine = AminoAcid("Tyrosine", "TYR", "Y",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=8,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=1
)

valine = AminoAcid("Valine", "VAL", "V",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=3,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

selenocysteine = AminoAcid("Selenocysteine", "SEC", "U",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2,
)

pyrrolysine = AminoAcid("Pyrrolysine", "PYL", "O",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=13,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4
)

alysine = AminoAcid("Alysine", "ALY", "K",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=13,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4
)

methyllysine = AminoAcid("Methyllysine", "MLZ", "K",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=14,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4
)

dimethyllysine = AminoAcid("Dimethyllysine", "MLY", "K",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=15,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4
)

trimethyllysine = AminoAcid("Trimethyllysine", "3ML", "K",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=16,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4
)

epsilon_methionine = AminoAcid("Epsilon Methionine", "MSE", "M",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

hydroxy_proline = AminoAcid("Hydroxy Proline", "HYP", "P",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.POLAR,
    size=4,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2
)

serine_phosphate = AminoAcid("Serine Phosphate", "SEP", "S",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.NEGATIVE_CHARGE,
    size=7,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4
)

threonine_phosphate = AminoAcid("Threonine Phosphate", "TOP", "T",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.NEGATIVE_CHARGE,
    size=8,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4
)

tyrosine_phosphate = AminoAcid("Tyrosine Phosphate", "TYP", "Y",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.NEGATIVE_CHARGE,
    size=13,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4
)

tyrosine_sulphate = AminoAcid("Tyrosine Sulphate", "TYS", "Y",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.NEGATIVE_CHARGE,
    size=13,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4
)

cyclohexane_alanine = AminoAcid("Cyclohexane Alanine", "CHX", "?",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.APOLAR,
    size=7,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)

unknown_amino_acid = AminoAcid("Unknown", "XXX", "X",
    charge=0.0,  # unknown, but needs a numerical value
    polarity=Polarity.APOLAR,
    size=0,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0
)


amino_acids = [alanine, arginine, asparagine, aspartate, cysteine, glutamate, glutamine, glycine,
               histidine, isoleucine, leucine, lysine, methionine, phenylalanine, proline, serine,
               threonine, tryptophan, tyrosine, valine]
