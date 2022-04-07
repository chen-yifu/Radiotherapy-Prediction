import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from utils.printers import *
def engineer_features(df, df_metadata):
    # Construct new columns from existing data

    # Construct "age_at_dx" to be the age of the patient at the time of diagnosis
    age_at_dx = []
    for i, row in tqdm(df.iterrows()):
        dob = pd.to_datetime(row["PRE_dob"])
        dx_date = pd.to_datetime(row["PRE_dximg_date"])
        if str(dx_date) == "nan":
            age_at_dx.append(np.nan)
        else:
            years_elapsed = (dx_date - dob).days / 365.25
            age_at_dx.append(years_elapsed)
    
    my_print("✅  Feature Engineering")
    
