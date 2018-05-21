
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
import pprint

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

## Temp:
core = 0
model = "PBN1"
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

#---------------------------------------------Development----------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower() + "/" + structure)

# Loading stan fit object and model:
with open("model_fit.pkl", "rb") as f:
    data_dict = pickle.load(f)

# Indexing the fit object and model
fit = data_dict['fit']

# Extracting the outputs:
outs = fit['trn'].extract()

# Getting the predictions:
if bool(re.search('-', structure)):
	y_pred = dict()
	for i in range(len(struc)):
		# Getting posterior mean of the parameters:
		mu_tmp = outs['mu_' + str(i)].mean(axis=0)
		beta_tmp = outs['beta_' + str(i)].mean(axis=0)
		alpha_tmp = outs['alpha_' + str(i)].mean(axis=0)
		if model == "PBN0":
			eta_tmp = outs['eta'].mean(axis=0)
		if model == "PBN1":
			eta_tmp = outs['eta_' + str(i)].mean(axis=0)
		# Computing predictions for train, dev, and test sets:
		y_pred[struc[i] + '_trn'] = mu_tmp + X[struc[i]]['nobin_trn'].dot(beta_tmp) + X[struc[i]]['bin_trn'].dot(alpha_tmp + eta_tmp)
		y_pred[struc[i] + '_dev'] = mu_tmp + X[struc[i]]['nobin_dev'].dot(beta_tmp) + X[struc[i]]['bin_dev'].dot(alpha_tmp + eta_tmp)
		y_pred[struc[i] + '_tst'] = mu_tmp + X[struc[i]]['nobin_tst'].dot(beta_tmp) + X[struc[i]]['bin_tst'].dot(alpha_tmp + eta_tmp)
else:
	# Getting the predictions:
	y_pred = dict()
	y_pred['trn'] = outs['mu'].mean(axis=0) + X['trn'].dot(outs['beta'].mean(axis=0))
	y_pred['dev'] = outs['mu'].mean(axis=0) + X['dev'].dot(outs['beta'].mean(axis=0))
	y_pred['tst'] = outs['mu'].mean(axis=0) + X['tst'].dot(outs['beta'].mean(axis=0))


# Computing metrics:
if bool(re.search('-', structure)):
	rmse_dict = dict()
	acc_dict = dict()
	r2_dict = dict()
	for i in list(y_pred.keys()):
		tmp = i.split('_')
		# Computing root mean squared error:
		rmse_dict[i] = round(rmse(y[tmp[0] + '_' + tmp[1]][tmp[2]].values.flatten(), y_pred[i]), 4)
		# Computing accuracy:
		acc_dict[i] = round(pearsonr(y[tmp[0] + '_' + tmp[1]][tmp[2]].values.flatten(), y_pred[i])[0], 4)
		# Computing r2 score:
		r2_dict[i] = round(r2_score(y[tmp[0] + '_' + tmp[1]][tmp[2]].values.flatten(), y_pred[i]), 4)
	# Printing results:
	pprint.pprint(rmse_dict)
	pprint.pprint(acc_dict)
	pprint.pprint(r2_dict)
else:
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


# Bar plot of the biomass bin effects:
pd.DataFrame(outs['alpha_0'].mean(axis=0), columns=['biomass']) \
    .plot.bar()
plt.show()
plt.clf()

# Bar plot of the height bin effects:
pd.DataFrame(outs['alpha_1'].mean(axis=0), columns=['height']) \
    .plot.bar()
plt.show()
plt.clf()

# Bar plot of the pleiotropyic bin effects:
pd.DataFrame(outs['eta_0'].mean(axis=0), columns=['eta']) \
    .plot.bar()
plt.show()
plt.clf()

# Bar plot of the pleiotropyic bin effects:
pd.DataFrame(np.abs(outs['z'].mean(axis=0)), columns=['eta']) \
    .plot.bar()
plt.show()
plt.clf()


# Bar plot of the pleiotropyic bin effects:
plt.bar(range(outs['z'].mean(axis=0).size), np.abs(outs['z'].mean(axis=0)))
plt.show()
plt.clf()

plt.bar(range(outs['eta_0'].mean(axis=0).size), np.abs(outs['eta_0'].mean(axis=0)))
plt.show()
plt.clf()

plt.bar(range(outs['eta_1'].mean(axis=0).size), np.abs(outs['eta_1'].mean(axis=0)))
plt.show()
plt.clf()

# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the genomic binned matrix under Cockerham's model:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Genetic correlation:
g0 = (outs['alpha_0'].mean(axis=0) + outs['eta_0'].mean(axis=0))
g1 = (outs['alpha_1'].mean(axis=0) + outs['eta_1'].mean(axis=0))
pearsonr(W_bin.dot(g0), W_bin.dot(g1))
plt.scatter(g0,g1)
plt.show()

