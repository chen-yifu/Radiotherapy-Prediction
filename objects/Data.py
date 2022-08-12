import pandas as pd 
import config

class Data:
    def __init__(self):
        metadata_path = config.metadata_path
        raw_df_path = config.raw_df_path
        processed_df_path = config.processed_df_path
        self.metadata = pd.read_excel(metadata_path)
        self.raw_df = pd.read_csv(raw_df_path) if raw_df_path else None
        self.processed_df = pd.read_csv(processed_df_path) if processed_df_path else None
        processed_PRE_df = self.processed_df.copy()
        for column in processed_PRE_df.columns:
            if column.startswith("POS_") or column.startswith("INT_"):
                processed_PRE_df.drop(column, axis=1, inplace=True)

        self.storage = {
            "metadata": self.metadata,
            "raw": self.raw_df,
            "processed": self.processed_df,
            "processed_PRE": processed_PRE_df            
        }
        

    def add_df(self, df: pd.DataFrame, key: str):
        if key in self.storage:
            print(f"Data key {key} already exists, overwriting with new df.")
        self.storage[key] = df
        
    def get_df(self, key: str):
        if key in self.storage:
            return self.storage[key]
        else:
            print(f"Data key {key} does not exist.")
            return None
        
    def list_df(self):
        dfs = list(self.storage.keys())
        print(dfs)
        return dfs
