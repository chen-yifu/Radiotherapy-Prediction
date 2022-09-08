import config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class Initializer:
    def __init__(self, metadata_path, raw_df_path=None, processed_df_path=None, DPI=150):
        config.metadata_path = self.metadata_path = metadata_path
        config.raw_df_path = self.raw_df_path = raw_df_path
        config.processed_df_path = self.processed_df_path = processed_df_path
        config.Initializer = self
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        plt.rcParams['figure.dpi'] = DPI
        config.plot_dpi = DPI
        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('font', serif='Helvetica Neue') 
        matplotlib.rc('text', usetex='false') 
        matplotlib.rcParams.update({'font.size': 11})