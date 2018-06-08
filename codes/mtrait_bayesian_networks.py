
#------------------------------------------------Modules-----------------------------------------------------#

# Loading libraries:
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

#---------------------------------------------Loading data---------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + "data/cross_validation")

# Reading adjusted means:
y = pd.read_csv("y_cv1_drymass_k0_trn.csv", index_col=0, header=None)

# Reading feature matrix:
X = pd.read_csv("x_cv1_drymass_k0_trn.csv", index_col=0)

