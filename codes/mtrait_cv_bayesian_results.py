
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
parser.add_argument("-d", "--data", dest = "data", default = "cv1_biomass_trn", help="Data set to be analysed")
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")

args = parser.parse_args()

#------------------------------------------Code parameters---------------------------------------------------#

# # Current core where the analysis is happening:
# core = args.core

# # Specifying the model
# model = args.model         # 'DBN' or 'BN' or 'BNP'

# # Type of cross-validation scheme:
# cv = args.cv

# # Choosing the data structure to be analysed:
# structure = args.data

# ## Temp:
# core = 0
# model = "BN"
# cv = "CV1"
# structure = "cv1_height"

# Seed to recover the analysis:
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

#---------------------------------------------Development----------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower() + "/" + structure)

# Loading stan fit object and model:
with open("model_fit.pkl", "rb") as f:
    data_dict = pickle.load(f)

# Indexing the fit object and model
model = data_dict['model']
fit = data_dict['fit']

# Extracting the outputs:
outs = fit['trn'].extract()

# Getting the predictions:
y_pred = dict()
y_pred['trn'] = outs['mu'].mean(axis=0) + X['trn'].dot(outs['beta'].mean(axis=0))
y_pred['dev'] = outs['mu'].mean(axis=0) + X['dev'].dot(outs['beta'].mean(axis=0))
y_pred['tst'] = outs['mu'].mean(axis=0) + X['tst'].dot(outs['beta'].mean(axis=0))

# Density plots of different data types:
sns.set_style('whitegrid')
ax = sns.kdeplot(y['trn'].values.flatten(), bw=0.5, label='train set', shade=True)
ax = sns.kdeplot(y['dev'].values.flatten(), bw=0.5, label='dev set', shade=True)
ax = sns.kdeplot(outs['y_gen'].mean(axis=0), bw=0.5, label='gen set', shade=True)
ax = sns.kdeplot(y_pred['dev'], bw=0.5, label='pred set', shade=True)
ax.set_title('Density of different data types')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.clf()

# Scatter plots of different data types:
tmp = dict()
tmp['trn'] = np.polyfit(y['trn'].values.flatten(), y_pred['trn'], 1)
tmp['dev'] = np.polyfit(y['dev'].values.flatten(), y_pred['dev'], 1)
tmp['tst'] = np.polyfit(y['tst'].values.flatten(), y_pred['tst'], 1)
plt.scatter(y['trn'].values.flatten(), y_pred['trn'], label="trn", alpha=0.3)
plt.plot(y['trn'].values.flatten(), tmp['trn'][0] * y['trn'].values.flatten() + tmp['trn'][1])
plt.scatter(y['dev'].values.flatten(), y_pred['dev'], label="dev", alpha=0.3)
plt.plot(y['dev'].values.flatten(), tmp['dev'][0] * y['dev'].values.flatten() + tmp['dev'][1])
plt.scatter(y['tst'].values.flatten(), y_pred['tst'], label="tst", alpha=0.3)
plt.plot(y['tst'].values.flatten(), tmp['tst'][0] * y['tst'].values.flatten() + tmp['tst'][1])
plt.legend()
plt.title('Scatter pattern of different data types')
plt.xlabel('Observed data')
plt.ylabel("Predicted data")
upper_bound = np.max([y['trn'].max(), y['dev'].max() ,y['tst'].max()])
plt.xlim(0, upper_bound + upper_bound * 0.1)
plt.ylim(0, upper_bound + upper_bound * 0.1)
plt.show()
plt.clf()

# Printing rMSE:
round(rmse(y['trn'].values.flatten(), y_pred['trn']), 4)
round(rmse(y['dev'].values.flatten(), y_pred['dev']), 4)
round(rmse(y['tst'].values.flatten(), y_pred['tst']), 4)

# Printing pearsonr:
round(pearsonr(y['trn'].values.flatten(), y_pred['trn'])[0], 4)
round(pearsonr(y['dev'].values.flatten(), y_pred['dev'])[0], 4)
round(pearsonr(y['tst'].values.flatten(), y_pred['tst'])[0], 4)

# Printing r2:
round(r2_score(y['trn'].values.flatten(), y_pred['trn']), 4)
round(r2_score(y['dev'].values.flatten(), y_pred['dev']), 4)
round(r2_score(y['tst'].values.flatten(), y_pred['tst']), 4)


