import json

class SubsetColumns:
    
    def __init__(self, all_columns, subset_col_path_json=None):
        """
        Args:
            all_columns (list): List of all columns.
        """
        self.all_columns = all_columns
        self.subset_col_path_json = subset_col_path_json
        if subset_col_path_json is None:
            self.subset_columns = {}
        else:
            self.subset_columns = self._load_subset_columns()
        
    def add_subset(self, subset_name, subset_columns):
        """
        Adds a subset of columns.
        
        Args:
            subset_name (str): Name of subset.
            subset_columns (list): List of columns in subset.
            
        Returns:
            subset of columns that exist in all_columns
        """
        # Check if subset already exists
        if subset_name in self.subset_columns:
            print(f"Subset {subset_name} already exists, so overwriting")
        # Check if subset columns are in all columns
        subset_columns_filtered = []
        for col in subset_columns:
            if col not in self.all_columns:
                print(f"Column {col} not in all columns, so skipping")
            else:
                subset_columns_filtered.append(col)
        
        self.subset_columns[subset_name] = subset_columns_filtered
        return subset_columns_filtered    
    
    def get_columns(self, subset_name):
        """
        Gets subset columns.
        
        Args:
            subset_name (str): Name of subset.
            
        Returns:
            subset_columns (list): List of columns in subset.
        """
        if subset_name in self.subset_columns:
            return self.subset_columns[subset_name]
        else:
            print(f"Subset {subset_name} does not exist")
            return None
    def _load_subset_columns(self):
        """
        Loads subset columns from json file.
        
        Returns:
            subset_columns (dict): Dictionary of subset columns.
        """
        with open(self.subset_col_path_json, "r") as f:
            subset_columns = json.load(f)
        return subset_columns
    
    def _save_subset_columns(self, path):
        """
        Saves subset columns to json file.
        """
        with open(path, "w") as f:
            json.dump(self.subset_columns, f, indent=4)
        print(f"Saved subset columns to {path}")