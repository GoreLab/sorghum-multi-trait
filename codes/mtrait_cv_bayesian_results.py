
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
# matplotlib.use('GTK') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import os

import pystan as ps
import time
import pickle
import re
import pprint
from funcy import project

from scipy.stats.stats import pearsonr
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
model = "DBN"
cv = "CV1"
structure = "cv1_height"
# structure = "cv1_biomass_drymass-cv1_height"

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

# Different types of cross-validation sets:
sets = ['trn', 'dev', 'tst']

for s in sets:
  if bool(re.search('PBN', model)):
    # Getting the traits info:
    struc = pd.Series(structure).str.split("-")[0]
    for i in range(len(struc)):
    	if s==sets[0]:
    		# Initializing dictionaries:
    		y[struc[i]] = dict()
    		X[struc[i]] = dict()
    	# Loading the data
    	X[struc[i]][s] = pd.read_csv('x_' + struc[i] + '_' + s + '.csv', header = 0, index_col=0)
    	y[struc[i]][s] = pd.read_csv('y_' + struc[i] + '_' + s + '.csv', header = 0, index_col=0)
    	# Biomass:
    	if bool(re.search('biomass', struc[i])):
    	  # Subsetting just the desired factors:
    	  index = X[struc[i]][s].columns.str.contains('|'.join(['loc','year']))
    	  X[struc[i]]['nobin_' + s] = X[struc[i]][s].loc[:,index]
    	  index = X[struc[i]][s].columns.str.contains('bin')
    	  X[struc[i]]['bin_' + s] = X[struc[i]][s].loc[:,index]
    	# Height:
    	if struc[i]=="cv1_height":
    	  # Subsetting just the desired factors:
    	  index = X[struc[i]][s].columns.str.contains('|'.join(['loc', 'year', 'dap']))
    	  X[struc[i]]['nobin_' + s] = X[struc[i]][s].loc[:,index]
    	  index = X[struc[i]][s].columns.str.contains('bin')
    	  X[struc[i]]['bin_' + s] = X[struc[i]][s].loc[:,index]
  else:
    # Loading the data:
    X[s] = pd.read_csv('x_' + structure + '_' + s + '.csv', header = 0, index_col=0)
    y[s] = pd.read_csv('y_' + structure + '_' + s + '.csv', header = 0, index_col=0)
    if structure=="cv1_biomass":
      # Subsetting just the desired factors:
      index = X[s].columns.str.contains('|'.join(['loc','year', 'bin']))
      X[s] = X[s].loc[:,index]
    if structure=="cv1_height":
      # Subsetting just the desired factors:
      index = X[s].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
      X[s] = X[s].loc[:,index]

# Preparing training data for the DBN Bayesian Network:
if model == 'DBN':
  # DAP indexes:
  tmp = df.dap.drop_duplicates()[1::] \
        .astype(int) \
        .astype(str) \
        .tolist()
  # Subsetting just the desired factors:
  for s in sets:
  	for t in tmp:
  	  subset = X[s]['dap'] == float(t)
  	  index = X[s].columns.str.contains('|'.join(['loc', 'year', 'dap']))
  	  X[t + '_nobin_' + s] = X[s].loc[subset,index]
  	  index = X[s].columns.str.contains('bin')
  	  X[t + '_bin_' + s] = X[s].loc[subset,index]
  	  y[t + '_' + s] = y[s][subset]


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
if model=='BN':
	# Getting the predictions:
	y_pred = dict()
	y_pred['trn'] = outs['mu'].mean(axis=0) + X['trn'].dot(outs['beta'].mean(axis=0))
	y_pred['dev'] = outs['mu'].mean(axis=0) + X['dev'].dot(outs['beta'].mean(axis=0))
	y_pred['tst'] = outs['mu'].mean(axis=0) + X['tst'].dot(outs['beta'].mean(axis=0))

if bool(re.search('PBN', model)):
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

if model=='DBN':
	# Getting the predictions:
	y_pred = dict()
	# DAP indexes:
	tmp = df.dap.drop_duplicates()[1::] \
	      .astype(int) \
	      .astype(str) \
	      .tolist()
	for t in range(len(tmp)):
		# Getting posterior mean of the parameters:
		mu_tmp = outs['mu_' + str(t)].mean(axis=0)
		beta_tmp = outs['beta_' + str(t)].mean(axis=0)
		alpha_tmp = outs['alpha_' + str(t)].mean(axis=0)
		d_tmp = outs['d'].mean(axis=0)
		# Computing predictions for train, dev, and test sets:
		y_pred[tmp[t] + '_trn'] = mu_tmp + X[tmp[t] + '_nobin_trn']['dap'] * d_tmp + X[tmp[t] + '_nobin_trn'].drop('dap',axis=1).dot(beta_tmp) + X[tmp[t] + '_bin_trn'].dot(alpha_tmp)
		y_pred[tmp[t] + '_dev'] = mu_tmp + X[tmp[t] + '_nobin_dev']['dap'] * d_tmp + X[tmp[t] + '_nobin_dev'].drop('dap',axis=1).dot(beta_tmp) + X[tmp[t] + '_bin_dev'].dot(alpha_tmp)
		y_pred[tmp[t] + '_tst'] = mu_tmp + X[tmp[t] + '_nobin_tst']['dap'] * d_tmp + X[tmp[t] + '_nobin_tst'].drop('dap',axis=1).dot(beta_tmp) + X[tmp[t] + '_bin_tst'].dot(alpha_tmp)

# Computing metrics:
if model=='BN':
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

if bool(re.search('PBN', model)):
	rmse_dict = dict()
	acc_dict = dict()
	r2_dict = dict()
	for i in list(y_pred.keys()):
		tmp = i.split('_')
		if bool(re.search('biomass', i)):
			# Computing root mean squared error:
			rmse_dict[i] = round(rmse(y[tmp[0] + '_' + tmp[1] + '_' + tmp[2]][tmp[3]].values.flatten(), y_pred[i]), 4)
			# Computing accuracy:
			acc_dict[i] = round(pearsonr(y[tmp[0] + '_' + tmp[1] + '_' + tmp[2]][tmp[3]].values.flatten(), y_pred[i])[0], 4)
			# Computing r2 score:
			r2_dict[i] = round(r2_score(y[tmp[0] + '_' + tmp[1] + '_' + tmp[2]][tmp[3]].values.flatten(), y_pred[i]), 4)
		else:
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

if model=='DBN':
	for s in ['trn', 'dev', 'tst']:
		# Index for subsetting the dictionary:
		index = ['30_' + s, '45_' + s, '60_' + s, '75_' + s, '90_' + s, '105_' + s, '120_' + s]
		# Subsettting items from the dictionary:
		tmp = project(y, index)
		# Stacking prediction and observations observed over time:
		y_obs_tmp = np.concatenate([tmp[x] for x in tmp], 0)
		tmp = project(y_pred, index)
		y_pred_tmp = np.concatenate([tmp[x] for x in tmp], 0)
		# Printing rMSE:
		round(rmse(y_obs_tmp, y_pred_tmp), 4)
		# # Printing pearsonr:
		# round(pearsonr(y_obs_tmp, y_pred_tmp)[0], 4)
		# # Printing r2:
		# round(r2_score(y_obs_tmp, y_pred_tmp), 4)

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
plt.bar(range(outs['z'].mean(axis=0).size), outs['z'].mean(axis=0))
plt.show()
plt.clf()

plt.bar(range(outs['eta_0'].mean(axis=0).size), outs['eta_0'].mean(axis=0))
plt.show()
plt.clf()

plt.bar(range(outs['eta_1'].mean(axis=0).size), outs['eta_1'].mean(axis=0))
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






lamb=2.5

indicator = np.empty(outs['z'].shape)
indicator[:] = np.nan

for s in range(indicator.shape[0]):
	indicator[s,:] = (outs['z'][s,:] > lamb) | (outs['z'][s,:] < -1 * lamb)

prob = indicator.mean(axis=0)

plt.bar(range(prob.size), prob)
plt.show()
plt.clf()




