import config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
class Initializer:
    def __init__(
        self, 
        metadata_path, 
        raw_df_path, 
        results_dir,
        processed_df_path=None, 
        DPI=150, 
        models_to_show=[]
        ):
        
        config.metadata_path = self.metadata_path = metadata_path
        config.raw_df_path = self.raw_df_path = raw_df_path
        config.results_dir = self.results_dir = results_dir
        config.processed_df_path = self.processed_df_path = processed_df_path
        config.Initializer = self
        config.plot_dpi = DPI
        config.models_to_show = models_to_show
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        matplotlib.rcParams.update({'font.size': 11})
        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('font', serif='Helvetica Neue') 
        matplotlib.rc('text', usetex='false') 
        plt.rcParams['figure.dpi'] = DPI