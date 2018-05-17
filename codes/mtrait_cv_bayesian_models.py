
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
import re

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
model = args.model         # 'DBN' or 'BN' or 'PBN'

# Type of cross-validation scheme:
cv = args.cv

## Temp:
core = 0
model = "PBN"  
cv = "CV1"
structure = "cv1_biomass-cv1_height"

# Seed to recover the analysis:
seed = core


#--------------------------------------------Reading data----------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + 'data/cross_validation/' + cv.lower())

# Initialize list to receive the outputs:
y = dict()
X = dict()

if bool(re.search('-', structure)):
  # Getting the traits info:
  struc = pd.Series(structure).str.split("-")[0]
  for i in range(len(struc)):
    # Initializing dictionaries:
    y[struc[i]] = dict()
    X[struc[i]] = dict()
    # Loading the data
    X[struc[i]]['trn'] = pd.read_csv('x_' + struc[i] + '_trn.csv', header = 0, index_col=0)
    y[struc[i]]['trn'] = pd.read_csv('y_' + struc[i] + '_trn.csv', header = 0, index_col=0)
    X[struc[i]]['dev'] = pd.read_csv('x_' + struc[i] + '_dev.csv', header = 0, index_col=0)
    y[struc[i]]['dev'] = pd.read_csv('y_' + struc[i] + '_dev.csv', header = 0, index_col=0)
    X[struc[i]]['tst'] = pd.read_csv('x_' + struc[i] + '_tst.csv', header = 0, index_col=0)
    y[struc[i]]['tst'] = pd.read_csv('y_' + struc[i] + '_tst.csv', header = 0, index_col=0)
    # Biomass:
    if struc[i]=="cv1_biomass":
      # Subsetting just the desired factors:
      index = X[struc[i]]['trn'].columns.str.contains('|'.join(['loc','year']))
      X[struc[i]]['nobin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['trn'].columns.str.contains('bin')
      X[struc[i]]['bin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['dev'].columns.str.contains('|'.join(['loc','year']))
      X[struc[i]]['nobin_dev'] = X[struc[i]]['dev'].loc[:,index]
      index = X[struc[i]]['dev'].columns.str.contains('bin')
      X[struc[i]]['bin_dev'] = X[struc[i]]['dev'].loc[:,index]
      index = X[struc[i]]['tst'].columns.str.contains('|'.join(['loc','year']))
      X[struc[i]]['nobin_tst'] = X[struc[i]]['tst'].loc[:,index]
      index = X[struc[i]]['tst'].columns.str.contains('bin')
      X[struc[i]]['bin_tst'] = X[struc[i]]['tst'].loc[:,index]
    # Height:
    if struc[i]=="cv1_height":
      # Subsetting just the desired factors:
      index = X[struc[i]]['trn'].columns.str.contains('|'.join(['loc', 'year', 'dap']))
      X[struc[i]]['nobin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['trn'].columns.str.contains('bin')
      X[struc[i]]['bin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['dev'].columns.str.contains('|'.join(['loc', 'year', 'dap']))
      X[struc[i]]['nobin_dev'] = X[struc[i]]['dev'].loc[:,index]
      index = X[struc[i]]['dev'].columns.str.contains('bin')
      X[struc[i]]['bin_dev'] = X[struc[i]]['dev'].loc[:,index]
      index = X[struc[i]]['tst'].columns.str.contains('|'.join(['loc', 'year', 'dap']))
      X[struc[i]]['nobin_tst'] = X[struc[i]]['tst'].loc[:,index]
      index = X[struc[i]]['tst'].columns.str.contains('bin')
      X[struc[i]]['bin_tst'] = X[struc[i]]['tst'].loc[:,index]
else:
  # Loading the data:
  X['trn'] = pd.read_csv('x_' + structure + '_trn.csv', header = 0, index_col=0)
  y['trn'] = pd.read_csv('y_' + structure + '_trn.csv', header = 0, index_col=0)
  X['dev'] = pd.read_csv('x_' + structure + '_dev.csv', header = 0, index_col=0)
  y['dev'] = pd.read_csv('y_' + structure + '_dev.csv', header = 0, index_col=0)
  X['tst'] = pd.read_csv('x_' + structure + '_tst.csv', header = 0, index_col=0)
  y['tst'] = pd.read_csv('y_' + structure + '_tst.csv', header = 0, index_col=0)  
  if structure=="cv1_biomass":
    # Subsetting just the desired factors:
    index = X['trn'].columns.str.contains('|'.join(['loc','year', 'bin']))
    X['trn'] = X['trn'].loc[:,index]
    index = X['dev'].columns.str.contains('|'.join(['loc','year', 'bin']))
    X['dev'] = X['dev'].loc[:,index]
    index = X['tst'].columns.str.contains('|'.join(['loc','year', 'bin']))
    X['tst'] = X['tst'].loc[:,index]
  if structure=="cv1_height":
    # Subsetting just the desired factors:
    index = X['trn'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
    X['trn'] = X['trn'].loc[:,index]
    index = X['dev'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
    X['dev'] = X['dev'].loc[:,index]
    index = X['tst'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
    X['tst'] = X['tst'].loc[:,index]


#----------------------------------------Bayesian Network code-----------------------------------------------#


if model=="BN":
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

if model=="PBN":
  # Initializing object to receive index for specific priors:
  index_x = dict()
  for i in range(len(struc)):
    # Index for specific priors:
    index_x[struc[i]] = dict()
    # Getting the features names prefix:
    tmp = X[struc[i]]['nobin_trn'].columns.str.split('_').str.get(0)
    # Building an incidence vector for adding specific priors for each feature class:
    index_x[struc[i]] = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 
    # Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
    index = X[struc[i]]['nobin_trn'].columns.str.contains('year')
    X[struc[i]]['year'] = X[struc[i]]['nobin_trn'].loc[:,index]
  # Storing all the data into a dictionary for pystan:
  df_stan = dict(n_0 = X[struc[0]]['nobin_trn'].shape[0],
                 p_x_0 = X[struc[0]]['nobin_trn'].shape[1],
                 p_z = X[struc[0]]['bin_trn'].shape[1],               
                 p_i_0 = np.max(index_x[struc[0]]),
                 p_r_0 = X[struc[0]]['year'].shape[1],
                 phi_0 = np.max(y[struc[0]]['trn'].values.flatten())*10,
                 index_x_0 = index_x[struc[0]],
                 X_0 = X[struc[0]]['nobin_trn'],
                 Z_0 = X[struc[0]]['bin_trn'],
                 X_r_0 = X[struc[0]]['year'],
                 y_0 = y[struc[0]]['trn'].values.flatten(),
                 n_1 = X[struc[1]]['nobin_trn'].shape[0],
                 p_x_1 = X[struc[1]]['nobin_trn'].shape[1],
                 p_i_1 = np.max(index_x[struc[1]]),
                 p_r_1 = X[struc[1]]['year'].shape[1],
                 phi_1 = np.max(y[struc[1]]['trn'].values.flatten())*10,
                 index_x_1 = index_x[struc[1]],
                 X_1 = X[struc[1]]['nobin_trn'],
                 Z_1 = X[struc[1]]['bin_trn'],
                 X_r_1 = X[struc[1]]['year'],
                 y_1 = y[struc[1]]['trn'].values.flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

if model=="BN":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='multi_trait.stan')

if model=="PBN":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='pleiotropic_multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['trn'] = model_stan.sampling(data=df_stan, chains=1, iter=400, seed=seed)

#-----------------------------------------Saving stan outputs------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower() + "/" + structure)

# Saving stan fit object and model:
with open("model_fit.pkl", "wb") as f:
    pickle.dump({'model' : model_stan, 'fit' : fit}, f, protocol=-1)

