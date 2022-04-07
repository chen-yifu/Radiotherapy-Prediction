import pandas as pd
from sklearn.impute import *
import numpy as np
from collections import Counter
from src.preprocess import preprocess
from utils.setup import *

# p = Printer()
setup()
preprocess.preprocess()