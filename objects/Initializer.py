import config
import pandas as pd
import matplotlib.pyplot as plt

class Initializer:
    def __init__(self, metadata_path, raw_df_path=None, processed_df_path=None):
        config.metadata_path = self.metadata_path = metadata_path
        config.raw_df_path = self.raw_df_path = raw_df_path
        config.processed_df_path = self.processed_df_path = processed_df_path
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        plt.rcParams['figure.dpi'] = 120
