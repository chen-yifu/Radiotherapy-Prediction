import pandas as pd
import config
import re
from collections import OrderedDict

class VarReader:
    
    def __init__(self):
        self.dictionary_path = config.dictionary_path
        self.dictionary = pd.read_excel(self.dictionary_path)

    def read_var_attrib(self, col_name, has_missing):
        try:
            definition = self.dictionary[self.dictionary["Field"] == col_name]
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
            print(e)
            return None

        
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

