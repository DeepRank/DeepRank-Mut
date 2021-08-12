import pandas

from deeprank.models.residue import Residue
from deeprank.config.chemicals import AA_codes_1to3


# >>> pandas.read_hdf("GPCR_variantsv2_increased_coverage.hdf5", "conservation")
#            sequence_residue_number amino_acid  alignment_position  alignment_name  sub_sequencecount  ...  sub_consv_W  sub_consv_X  sub_consv_Y  sub_consv_Z  sub_consv_gap
# accession                                                                                             ...                                                                   
# 4DAJC                            1          T                 NaN             NaN                NaN  ...          NaN          NaN          NaN          NaN            NaN
# 4DAJC                            2          I               101.0  sf_v2.adh_gpcr             1337.0  ...     0.005234          0.0     0.005985          0.0       0.225830
# 4DAJC                            3          W               102.0  sf_v2.adh_gpcr             1357.0  ...     0.510742          0.0     0.067810          0.0       0.214233
# 4DAJC                            4          Q               103.0  sf_v2.adh_gpcr             1360.0  ...     0.090454          0.0     0.001471          0.0       0.212524
# 4DAJC                            5          V               104.0  sf_v2.adh_gpcr             1369.0  ...     0.000000          0.0     0.002192          0.0       0.207275
# ...                            ...        ...                 ...             ...                ...  ...          ...          ...          ...          ...            ...
# 4N6HA                          404          R               375.0  sf_v2.adh_gpcr             3889.0  ...     0.032135          0.0     0.003857          0.0       0.432373
# 4N6HA                          405          K               263.0   pdb_v3.016122             1583.0  ...     0.001264          0.0     0.008209          0.0       0.638184
# 4N6HA                          406          P               264.0   pdb_v3.016122             1586.0  ...     0.009460          0.0     0.010719          0.0       0.637207
# 4N6HA                          407          C               265.0   pdb_v3.016122             1584.0  ...     0.030304          0.0     0.118713          0.0       0.637695
# 4N6HA                          408          G               266.0   pdb_v3.016122             1584.0  ...     0.084595          0.0     0.000631          0.0       0.637695


# >>> pandas.read_hdf("GPCR_variantsv2_increased_coverage.hdf5", "pdbs")
#           pdbnumber      pdb_x      pdb_y      pdb_z  pdb_accessibility  pdb_bvalue  pdb_residuenumber  alignment_position
# accession
# 5TZYA             2  -8.789062   8.476562  65.312500          154.37500    133.2500                1.0                 NaN
# 5TZYA             3 -10.773438  10.515625  62.750000           37.46875    196.7500                2.0                 NaN
# 5TZYA             4 -11.843750  14.164062  63.500000           70.00000    216.2500                3.0                 NaN
# 5TZYA             5 -15.531250  15.117188  62.843750           64.87500    201.8750                4.0                 NaN
# 5TZYA             6 -14.312500  18.312500  61.031250          112.81250     62.6875                5.0                 NaN
# ...             ...        ...        ...        ...                ...         ...                ...                 ...
# 6IGLA           398  30.125000  54.125000   7.976562          105.50000     82.3125              465.0               230.0
# 6IGLA           399  27.562500  54.968750  10.679688           69.43750     82.8125              466.0               231.0
# 6IGLA           400  26.687500  51.312500  11.312500           26.21875     81.8125              467.0               232.0
# 6IGLA           401  30.375000  50.312500  11.453125          115.00000     87.8750              468.0               233.0
# 6IGLA           402  31.484375  53.031250  13.882812          169.25000    100.8125              469.0               234.0


def get_conservation_from_bioprodict(pdb_dataframe, conservation_dataframe, pdb_accession, pdb_chain_id):
    accession_code = ("%s%s" % (pdb_accession, pdb_chain_id)).upper()

    pdb_translation = {}
    for index, row in pdb_dataframe.loc[accession_code].iterrows():
        pdb_number = row["pdbnumber"]
        seq_number = row["pdb_residuenumber"]
        pdb_translation[seq_number] = pdb_number

    conservation_table = {}
    for index, row in conservation_dataframe.loc[accession_code].iterrows():
        sequence_number = row["sequence_residue_number"]
        amino_acid = AA_codes_1to3[row["amino_acid"]]
        pdb_number = pdb_translation[sequence_number]

        residue_id = Residue(int(pdb_number), amino_acid, pdb_chain_id)
        conservation_table[residue_id] = {}
        for amino_acid_letter in AA_codes_1to3:
            key = "sub_consv_%s" % amino_acid_letter
            conservation_table[residue_id][amino_acid_letter] = row[key]

    return conservation_table

