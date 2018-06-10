
#------------------------------------------------Modules-----------------------------------------------------#

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Loading libraries:
import pandas as pd
import numpy as np
import os
import pystan as ps

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
y = pd.read_csv("y_cv1_height_k0_trn.csv", index_col=0)

# Reading feature matrix:
X = pd.read_csv("x_cv1_height_k0_trn.csv", index_col=0)


#------------------------------------------Data input for stan-----------------------------------------------#

groups = 

dict_stan = dict(n = X.shape[0],
				 p_x = 1,
				 p_z = (X.shape[0] - 1),


				) 

# To do list:
# - Remove heterogeneity of variance into the BN and PBN
# - Add heterogeneity of variance into the DBN
# - Change the code to run serial analysis for each DAP measure for BN and PBN
# - Prepare data input for stan for all models

#--------------------------------------Running the Bayesian Network------------------------------------------#


# Setting directory:
os.chdir(prefix_proj + "codes")

model_stan = ps.StanModel(file='bayesian_network.stan')

model_stan = ps.StanModel(file='pleiotropic_bayesian_network.stan')

model_stan = ps.StanModel(file='dynamic_bayesian_network_0_6.stan')
