
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
# matplotlib.use('GTK') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns

import os

import tensorflow as tf
import pystan as ps
import subprocess
import dill
import time
import sys
import pickle

from scipy.stats import skew
from scipy.stats import moment
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score

# For adding flags to the code:
import argparse
parser = argparse.ArgumentParser()

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documents/sorghum-multi-trait/"
# prefix_proj = "/home/jhonathan/Documentos/sorghum-multi-trait/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documents/resul_mtrait-proj/"
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

#-----------------------------------------Adding flags to the code-------------------------------------------#

# Getting flags:
parser.add_argument("-c", "--core", dest = "core", default = 0, help="Current core where the analysis is happing", type=int)
parser.add_argument("-d", "--data", dest = "data", default = "cv1_biomass", help="Data set to be analysed")
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")

args = parser.parse_args()

#------------------------------------------Code parameters---------------------------------------------------#

# Current core where the analysis is happening:
core = args.core

# Choosing the data structure to be analysed:
structure = args.data

# Specifying the model
model = args.model         # 'DBN' or 'BN' or 'BNP'

# Type of cross-validation scheme:
cv = args.cv

# ## Temp:
# core = 0
# model = "BN"
# cv = "CV1"
# structure = "cv1_biomass"

# # Seed to recover the analysis:
seed = core

#--------------------------------------------Reading data----------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + 'data/cross_validation/' + cv.lower())

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Loading the data:
X['trn'] = pd.read_csv('x_' + structure + '_trn.csv', header = 0, index_col=0)
y['trn'] = pd.read_csv('y_' + structure + '_trn.csv', header = 0, index_col=0)
X['dev'] = pd.read_csv('x_' + structure + '_dev.csv', header = 0, index_col=0)
y['dev'] = pd.read_csv('y_' + structure + '_dev.csv', header = 0, index_col=0)
X['tst'] = pd.read_csv('x_' + structure + '_tst.csv', header = 0, index_col=0)
y['tst'] = pd.read_csv('y_' + structure + '_tst.csv', header = 0, index_col=0)

# Subsetting just the desired factors:
index = X['trn'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['trn'] = X['trn'].loc[:,index]
index = X['dev'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['dev'] = X['dev'].loc[:,index]
index = X['tst'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['tst'] = X['tst'].loc[:,index]

#----------------------------------------Bayesian Network code-----------------------------------------------#

# Getting the features names prefix:
tmp = X['trn'].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
index = X['trn'].columns.str.contains('year')
X['year'] = X['trn'].loc[:,index]

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['trn'].shape[0],
               p_x = X['trn'].shape[1],
               p_i = np.max(index_x),
               p_r = X['year'].shape[1],
               phi = np.max(y['trn'].values.flatten())*10,
               index_x = index_x,
               X = X['trn'],
               X_r = X['year'],
               y = y['trn'].values.flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['trn'] = model.sampling(data=df_stan, chains=1, iter=400, seed=seed)

#-----------------------------------------Saving stan outputs------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower() + "/" + structure)

# Saving stan fit object and model:
with open("model_fit.pkl", "wb") as f:
    pickle.dump({'model' : model, 'fit' : fit}, f, protocol=-1)
