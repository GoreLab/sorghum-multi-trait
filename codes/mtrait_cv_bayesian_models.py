
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import os

import pystan as ps
import time
import pickle
import re

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

# ## Temp:
core = 0
model = "DBN"  
cv = "CV1"
structure = "cv1_height"
# structure = "cv1_biomass_drymass-cv1_height"
# structure = "cv1_biomass_starch-cv1_height"

# Seed to recover the analysis:
seed = core

#--------------------------------------------Reading data----------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the data frame with phenotypic data and id's:
df = pd.read_csv("df.csv", header = 0, index_col=0)

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
    # Biomass:
    if bool(re.search('biomass', struc[i])):
      # Subsetting just the desired factors:
      index = X[struc[i]]['trn'].columns.str.contains('|'.join(['loc','year']))
      X[struc[i]]['nobin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['trn'].columns.str.contains('bin')
      X[struc[i]]['bin_trn'] = X[struc[i]]['trn'].loc[:,index]
    # Height:
    if struc[i]=="cv1_height":
      # Subsetting just the desired factors:
      index = X[struc[i]]['trn'].columns.str.contains('|'.join(['loc', 'year', 'dap']))
      X[struc[i]]['nobin_trn'] = X[struc[i]]['trn'].loc[:,index]
      index = X[struc[i]]['trn'].columns.str.contains('bin')
      X[struc[i]]['bin_trn'] = X[struc[i]]['trn'].loc[:,index]
else:
  # Loading the data:
  X['trn'] = pd.read_csv('x_' + structure + '_trn.csv', header = 0, index_col=0)
  y['trn'] = pd.read_csv('y_' + structure + '_trn.csv', header = 0, index_col=0)
  if structure=="cv1_biomass":
    # Subsetting just the desired factors:
    index = X['trn'].columns.str.contains('|'.join(['loc','year', 'bin']))
    X['trn'] = X['trn'].loc[:,index]
  if structure=="cv1_height":
    # Subsetting just the desired factors:
    index = X['trn'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
    X['trn'] = X['trn'].loc[:,index]

# Preparing training data for the DBN Bayesian Network:
if model == 'DBN':
  # DAP indexes:
  tmp = df.dap.drop_duplicates()[1::] \
        .astype(int) \
        .astype(str) \
        .tolist() 
  # Subsetting just the desired factors:
  for t in tmp:
    subset = X['trn']['dap'] == float(t)
    index = X['trn'].columns.str.contains('|'.join(['loc', 'year', 'dap']))
    X[t + '_nobin_trn'] = X['trn'].loc[subset,index]
    index = X['trn'].columns.str.contains('bin')
    X[t + '_bin_trn'] = X['trn'].loc[subset,index]
    y[t + '_trn'] = y['trn'][subset]


#----------------------------------------Bayesian Network code-----------------------------------------------#

if model=="BN":
  # Getting the features names prefix:
  tmp = X['trn'].columns.str.split('_').str.get(0)
  # Building an incidence vector for adding specific priors for each feature class:
  index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 
  # Building an year matrix just for indexing resuduals standard deviations heterogeneous across years:
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

if bool(re.search('PBN', model)):
  # Initializing object to receive index for specific priors:
  index_x = dict()
  for i in range(len(struc)):
    # Index for specific priors:
    index_x[struc[i]] = dict()
    # Getting the features names prefix:
    tmp = X[struc[i]]['nobin_trn'].columns.str.split('_').str.get(0)
    # Building an incidence vector for adding specific priors for each feature class:
    index_x[struc[i]] = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 
    # Building an year matrix just for indexing resuduals standard deviations heterogeneous across years:
    index = X[struc[i]]['nobin_trn'].columns.str.contains('year')
    X[struc[i]]['year'] = X[struc[i]]['nobin_trn'].loc[:,index]
  # Storing all the data into a dictionary for pystan:
  df_stan = dict(n_0 = X[struc[0]]['nobin_trn'].shape[0],
                 p_x_0 = X[struc[0]]['nobin_trn'].shape[1],
                 p_z = X[struc[0]]['bin_trn'].shape[1],               
                 p_i_0 = np.max(index_x[struc[0]]),
                 p_r_0 = X[struc[0]]['year'].shape[1],
                 index_x_0 = index_x[struc[0]],
                 X_0 = X[struc[0]]['nobin_trn'],
                 Z_0 = X[struc[0]]['bin_trn'],
                 X_r_0 = X[struc[0]]['year'],
                 y_0 = y[struc[0]]['trn'].values.flatten(),
                 n_1 = X[struc[1]]['nobin_trn'].shape[0],
                 p_x_1 = X[struc[1]]['nobin_trn'].shape[1],
                 p_i_1 = np.max(index_x[struc[1]]),
                 p_r_1 = X[struc[1]]['year'].shape[1],
                 index_x_1 = index_x[struc[1]],
                 X_1 = X[struc[1]]['nobin_trn'],
                 Z_1 = X[struc[1]]['bin_trn'],
                 X_r_1 = X[struc[1]]['year'],
                 y_1 = y[struc[1]]['trn'].values.flatten(),
                 phi = np.max(y[struc[1]]['trn'].values.flatten())*10)


###############################


if model=='DBN':
  # DAP indexes:
  tmp = df.dap.drop_duplicates()[1::] \
        .astype(int) \
        .astype(str) \
        .tolist() 
  # Initializing object to receive index for specific priors:
  index_x = dict()
  for t in tmp:
    # Getting the features names prefix:
    subset = X[t + '_nobin_trn'].columns.str.split('_').str.get(0)
    # Building an incidence vector for adding specific priors for each feature class:
    index_x[t] = pd.DataFrame(subset).replace(subset.drop_duplicates(), range(1,(subset.drop_duplicates().size+1)))[0].values 
    # Building an year matrix just for indexing resuduals standard deviations heterogeneous across years:
    index = X[t + '_nobin_trn'].columns.str.contains('year')
    X[t + '_year'] = X[t + '_nobin_trn'].loc[:,index]
  if cv == 'CV1':
    df_stan = dict(n_0 = y['30_trn'].size,
                   p_x_0 = X['30_nobin_trn'].shape[1],
                   p_z_0 = X['30_bin_trn'].shape[1],
                   p_i_0 = np.max(index_x['30']),
                   p_r_0 = X['30_year'].shape[1],
                   index_x_0 = index_x['30'],
                   x_d_0 = X['30_nobin_trn']['dap'],   
                   X_0 = X['30_nobin_trn'].drop('dap',axis=1),
                   Z_0 = X['30_bin_trn'],
                   X_r_0 = X['30_year'],
                   y_0 = y['30_trn'].values.flatten(),
                   n_1 = y['45_trn'].size,
                   p_x_1 = X['45_nobin_trn'].shape[1],
                   p_z_1 = X['45_bin_trn'].shape[1],
                   p_i_1 = np.max(index_x['45']),
                   p_r_1 = X['45_year'].shape[1],
                   index_x_1 = index_x['45'],
                   x_d_1 = X['45_nobin_trn']['dap'],   
                   X_1 = X['45_nobin_trn'].drop('dap',axis=1),
                   Z_1 = X['45_bin_trn'],
                   X_r_1 = X['45_year'],
                   y_1 = y['45_trn'].values.flatten(),
                   n_2 = y['60_trn'].size,
                   p_x_2 = X['60_nobin_trn'].shape[1],
                   p_z_2 = X['60_bin_trn'].shape[1],
                   p_i_2 = np.max(index_x['60']),
                   p_r_2 = X['60_year'].shape[1],
                   index_x_2 = index_x['60'],
                   x_d_2 = X['60_nobin_trn']['dap'],   
                   X_2 = X['60_nobin_trn'].drop('dap',axis=1),
                   Z_2 = X['60_bin_trn'],
                   X_r_2 = X['60_year'],
                   y_2 = y['60_trn'].values.flatten(),
                   n_3 = y['75_trn'].size,
                   p_x_3 = X['75_nobin_trn'].shape[1],
                   p_z_3 = X['75_bin_trn'].shape[1],
                   p_i_3 = np.max(index_x['75']),
                   p_r_3 = X['75_year'].shape[1],
                   index_x_3 = index_x['75'],
                   x_d_3 = X['75_nobin_trn']['dap'],   
                   X_3 = X['75_nobin_trn'].drop('dap',axis=1),
                   Z_3 = X['75_bin_trn'],
                   X_r_3 = X['75_year'],
                   y_3 = y['75_trn'].values.flatten(),
                   n_4 = y['90_trn'].size,
                   p_x_4 = X['90_nobin_trn'].shape[1],
                   p_z_4 = X['90_bin_trn'].shape[1],
                   p_i_4 = np.max(index_x['90']),
                   p_r_4 = X['90_year'].shape[1],
                   index_x_4 = index_x['90'],
                   x_d_4 = X['90_nobin_trn']['dap'],   
                   X_4 = X['90_nobin_trn'].drop('dap',axis=1),
                   Z_4 = X['90_bin_trn'],
                   X_r_4 = X['90_year'],
                   y_4 = y['90_trn'].values.flatten(),
                   n_5 = y['105_trn'].size,
                   p_x_5 = X['105_nobin_trn'].shape[1],
                   p_z_5 = X['105_bin_trn'].shape[1],
                   p_i_5 = np.max(index_x['105']),
                   p_r_5 = X['105_year'].shape[1],
                   index_x_5 = index_x['105'],
                   x_d_5 = X['105_nobin_trn']['dap'],   
                   X_5 = X['105_nobin_trn'].drop('dap',axis=1),
                   Z_5 = X['105_bin_trn'],
                   X_r_5 = X['105_year'],
                   y_5 = y['105_trn'].values.flatten(),
                   n_6 = y['120_trn'].size,
                   p_x_6 = X['120_nobin_trn'].shape[1],
                   p_z_6 = X['120_bin_trn'].shape[1],
                   p_i_6 = np.max(index_x['120']),
                   p_r_6 = X['120_year'].shape[1],
                   index_x_6 = index_x['120'],
                   x_d_6 = X['120_nobin_trn']['dap'],   
                   X_6 = X['120_nobin_trn'].drop('dap',axis=1),
                   Z_6 = X['120_bin_trn'],
                   X_r_6 = X['120_year'],
                   y_6 = y['120_trn'].values.flatten(),
                   phi = np.max(y['120_trn'].values.flatten())*10)






###############################


# Setting directory:
os.chdir(prefix_proj + "codes")

if model=="BN":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='multi_trait.stan')

if model=="PBN0":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='pleiotropic_multi_trait_0.stan')

if model=="PBN1":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='pleiotropic_multi_trait_1.stan')

if model=="DBN":
  # Compiling the C++ code for the model:
  model_stan = ps.StanModel(file='dynamic_bayesian_network_0_6.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['trn'] = model_stan.sampling(data=df_stan, chains=4, iter=400, seed=seed)

#-----------------------------------------Saving stan outputs------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower() + "/" + structure)

# Saving stan fit object and model:
with open("model_fit.pkl", "wb") as f:
    pickle.dump({'model' : model_stan, 'fit' : fit}, f, protocol=-1)


