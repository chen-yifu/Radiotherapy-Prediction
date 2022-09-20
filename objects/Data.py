import pandas as pd 
import config

class Data:
    def __init__(self):
        metadata_path = config.metadata_path
        raw_df_path = config.raw_df_path
        processed_df_path = config.processed_df_path
        config.Data = self
        
        metadata = pd.read_excel(metadata_path)
        raw_df = pd.read_csv(raw_df_path).apply(pd.to_numeric, errors='coerce') if raw_df_path else pd.DataFrame()
        processed_df = pd.read_csv(processed_df_path).apply(pd.to_numeric, errors='coerce') if processed_df_path else pd.DataFrame()
        processed_PRE_df = processed_df.copy()
        
        for column in processed_PRE_df.columns:
            if column.startswith("POS_") or column.startswith("INT_"):
                processed_PRE_df.drop(column, axis=1, inplace=True)

        self.storage = {
            "metadata": metadata,
            "raw": raw_df,
            "processed": processed_df,
            "processed_PRE": processed_PRE_df            
        }
        
        self.metadata = metadata
        self.raw_df = raw_df
        self.processed_df = processed_df
        self.processed_PRE_df = processed_PRE_df
         

    def _format_name(self, name: str, is_standardized=False, is_PRE_only=False, is_PRE_and_POS=False, is_ready=False) -> str:
        """Format the name of a DataFrame.

        Args:
            name (str): Name of the DataFrame.
            is_standardized (bool, optional): Defaults to False. Whether the DataFrame is standardized.
            is_PRE_only (bool, optional): Defaults to False. Whether the DataFrame is PRE columns-only.
            is_PRE_and_POS (bool, optional): Defaults to False. Whether the DataFrame is PRE and POS columns-only.
            is_ready (bool, optional): Defaults to False. Whether the DataFrame is ready to be used for training/evaluation.

        Returns:
            str: _description_
        """
        if is_standardized:
            name = name + "_standardized"
        elif is_PRE_only:
            name = name + "_PRE_only"
        elif is_PRE_and_POS:
            name = name + "_PRE_and_POS"
        else:
            print(f"DataFrame name should be either standardized, PRE_only, or PRE_and_POS.")
        if is_ready:
            name = name + "_ready"
        return name
    
    
    def add_df(self, df: pd.DataFrame, name: str, is_standardized=False, is_PRE_only=False, is_PRE_and_POS=False, is_ready=False):
        """Add a DataFrame to the storage.

        Args:
            df (pd.DataFrame): DataFrame to add.
            name (str): Name of the DataFrame.
            is_standardized (bool, optional): Defaults to False. Whether the DataFrame is standardized.
            is_PRE_only (bool, optional): Defaults to False. Whether the DataFrame is PRE columns-only.
            is_PRE_and_POS (bool, optional): Defaults to False. Whether the DataFrame is PRE and POS columns-only.
            is_ready (bool, optional): Defaults to False. Whether the DataFrame is ready to be used for training/evaluation.
        """
        if name in self.storage:
            print(f"DataFrame named {name} already exists, overwriting with new DataFrame.")
        
        name = self._format_name(name, is_standardized, is_PRE_only, is_PRE_and_POS, is_ready)
            
        self.storage[name] = df
        print(f"DataFrame {name} with shape {df.shape} added to storage.")
        return name
        
        
    def get_df(self, name: str):
        """Get a DataFrame from the storage.

        Args:
            name (str): Name of the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the given name.
        """
        if name in self.storage:
            return self.storage[name]
        else:
            print(f"DataFrame named {name} does not exist.")
            return None


    def get_df_from_experiment_name(self, experiment_name: str, is_standardized=False, is_PRE_only=False, is_PRE_and_POS=False, is_ready=False):
        name = self._format_name(experiment_name, is_standardized, is_PRE_only, is_PRE_and_POS, is_ready)
        return self.get_df(name)

        
    def list_df(self):
        """List all DataFrames in the storage."""
        df_names = list(self.storage.keys())
        df_shapes = [df.shape for df in self.storage.values()]
        print(f"There are {len(df_names)} DataFrames in the storage: {df_names}.")
        return list(zip(df_names, df_shapes))


    def get_default_df(self):
        """Get the default DataFrame.

        Returns:
            pd.DataFrame: The default DataFrame.
        """
        return self.processed_df