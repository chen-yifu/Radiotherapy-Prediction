from src.preprocess import preprocess

metadata_path = "data/testing/input/metadata/Metadata.xlsx"
df_path = "data/testing/input/AllTranTrainVal.csv"


def load_df():
    preprocess.preprocess(
        experiment_dir="data/testing/output/2022-05-11-094013_test",
        very_solid_threshold=0.05,
        solid_threshold=0.20,
        metadata_path=metadata_path,
        df_path=df_path,
        use_solid_cols=True
    )
    
