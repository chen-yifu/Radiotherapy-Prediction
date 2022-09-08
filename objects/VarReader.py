import pandas as pd
import config
import re
from collections import OrderedDict

class VarReader:
    
    def __init__(self):
        self.metadata_path = config.metadata_path
        self.metadata = pd.read_excel(self.metadata_path)
        config.VarReader = self

    def read_var_attrib(self, col_name, has_missing):
        try:
            definition = self.metadata[self.metadata["Field"] == col_name]
            section = definition["Group"].iloc[0]
            if str(section) == "nan":
                section = "Other"
                print(f"{col_name} is not in the dictionary")
            section = section.upper()[:3]
            dtype = definition["Type"].iloc[0]  # radio, checkbox, text, yesno
            label = definition["Description"].iloc[0]
            options = definition["Values"].iloc[0]
            # Replace special characters and symbols in section and label using regex
            section = re.sub(r'[^\w]', ' ', section)
            # label = re.sub(r'[^\w]', ' ', label)
            # Options have format {tick_value: label, ...}
            if str(options) != "nan":
                    options = {float(x.split(",")[0].strip()) : x.split(",")[1].strip() for x in options.split("|")}
                    if has_missing:
                        options[-1] = "missing"
                    options = OrderedDict(sorted(options.items()))
            else:
                options = {}
            options_str = " | ".join([str(x) + ", " + str(y) for x, y in options.items()])
            return_dict = {
                "section": section,
                "dtype": dtype,
                "label": label,
                "options": options,
                "options_str": options_str
            }
            return return_dict
        except Exception as e:
            print(f"Error reading attributes for {col_name}")
            raise e

    def add_var(self, col_name, section, dtype, label, options):
        # Check if column already exists in metadata, if so, drop it in place
        if col_name in self.metadata["Field"].values:
            self.metadata.drop(self.metadata[self.metadata["Field"] == col_name].index, inplace=True)
        # Concatenate with original metadata
        new_row = {"Field": col_name, "Group": section, "Type": dtype, "Description": label, "Values": "|".join([str(k)+','+str(v) for k,v in options.items()])}
        self.metadata = pd.concat([self.metadata, pd.DataFrame(new_row, index=[0])])
        
        
    def has_missing(self, df, col_name):
        # Return true if any value in column is is -1
        for value in df[col_name]:
            if value == -1:
                return True
        # return true if the entire column is np.nan
        all_nan = df[col_name].isna().all()
        if all_nan:
            return True
        else:
            return False


    def is_dtype_categorical(self, dtype):
        if dtype in ["radio", "checkbox", "yesno", "categorical", "ordinal"]:
            return True
        elif dtype in ["numeric", "date", "datetime", "time", "real", "integer", "float", "text"]:
            return False
        else:
            raise ValueError(f"Unknown dtype: '{dtype}'.")