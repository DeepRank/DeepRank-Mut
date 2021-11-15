import pandas


CONSERVATIONS_DIRECTORY_PATH = None  # must be set before running


def get_c_alpha_pos(variant):
    db = pdb2sql(variant.pdb_path)
    try:
        position = db.get("x,y,z", chainID=variant.chain_id, resSeq=variant.residue_number, name="CA")[0]

        return position
    finally:
        db._close()


def get_conservations_dataframe(protein_ac):
    if CONSERVATIONS_DIRECTORY_PATH is None:
        raise RuntimeError("CONSERVATIONS_DIRECTORY_PATH is not set, please set it to a valid directory")

    return pandas.read_parquet(os.path.join(CONSERVATIONS_DIRECTORY_PATH, "{}.parq".format(protein_ac)))


WT_FEATURE_NAME = "wildtype_conservation"
VAR_FEATURE_NAME = "variant_conservation"


def __compute_feature__(pdb_data, feature_group, raw_feature_group, variant):
    "this feature module adds amino acid conservation as deeprank features"

    if variant.protein_ac is None or variant.protein_residue_number is None:
        raise RuntimeError("protein accession and protein residue number must be set on the variant to generate this feature")

    # Get conservations data from file:
    conservations_dataframe = get_conservations_dataframe(variant.protein_ac)
    residue_row = conservations_dataframe.iloc[variant.protein_residue_number]
    wildtype_conservation = residue_row["sub_consv_{}".format(variant.wildtype_amino_acid.letter)
    variant_conservation = residue_row["sub_consv_{}".format(variant.variant_amino_acid.letter)

    # Get the C-alpha position to store the feature with:
    c_alpha_position = get_c_alpha_pos(variant)
    xyz_key = tuple(c_alpha_position)

    # Store features:
    feature_object = FeatureClass("Residue")
    feature_object.feature_data_xyz[WT_FEATURE_NAME] = {xyz_key: [wildtype_conservation]}
    feature_object.feature_data_xyz[VAR_FEATURE_NAME = {xyz_key: [variant_conservation]}

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
