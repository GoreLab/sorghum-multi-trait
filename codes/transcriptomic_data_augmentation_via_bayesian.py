
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

from scipy.stats import skew
from scipy.stats import moment
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score
import statsmodels.api as sm

# For adding flags to the code:
import argparse
parser = argparse.ArgumentParser()

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs/rnaseq_imp will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

#-----------------------------------------Adding flags to the code-------------------------------------------#

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-c", "--core", help="Current core where the analysis is happing")
parser.add_argument("-nc", "--ncores", help="Number of cores")
parser.add_argument("-a", "--alt", help="Number of alternative models per bin")

args = parser.parse_args()

print( "ncores {} alt_bin {}".format(
        args.ncores,
        args.alt,
     ))

#----------------------------------Loading genomic and transcriptomic data-----------------------------------#


# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the data frame with phenotypic data and id's:
df = pd.read_csv("df.csv", header = 0, index_col=0)

# Loading the genomic binned matrix under Cockerham's model:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Loading the transcriptomic binned matrix:
T = pd.read_csv("T.csv", header = 0, index_col=0)

#----------------------------------Preparing data for cross-validation---------------------------------------#

## Temp:
r=200
core=0
n_cores=40
n_alt=200

# # Current core where the analysis is happening:
# core = parser.core

# # Number of cores to do the partition of the bins
# n_cores = parser.ncores

# # Number of alternative runs per bin:
# n_alt = parser.alt

# Seed to recover the analysis:
seed = int(str(core) + str(n_alt) + str(r))

# Splitting the index of the columns from the binned genomic matrix into subsets:
index_wbin = np.array_split(T.columns, n_cores)

# for r in range(index_wbin[core].size):


# Creating an empty dictionary to receive feature matrices, and responses:
X = dict()
y = dict()

# Indexing the phenotype (transcription at bin r)
y['full'] = T[index_wbin[core].values[r]]

# Feature matrix considering only individuals with genomic and transcriptomic data:
X['full'] = W_bin.loc[T.index]

# Indexing the genomic data of the real missing transcriptomic dat:
X['miss'] = W_bin.loc[np.invert(W_bin.index.isin(T.index))]
 
# Index to receive the position of the data frame:
index_cv = dict()

# Subsetting data into train and (dev set + test set) for height data:
X['trn'], X['dev'], y['trn'], y['dev'], index_cv['trn'], index_cv['dev'] = train_test_split(X['full'], 
			  																																								    y['full'],
 		                                                																				X['full'].index,
                                                        																		test_size=0.3,
                                                        																		random_state=seed)

# Subsetting (dev set + test set) into dev set and test set:
X['dev'], X['tst'], y['dev'], y['tst'], index_cv['dev'], index_cv['tst'] = train_test_split(X['dev'],
	                                                            		  												y['dev'],
	                                                            		  												index_cv['dev'],
                                                          				  												test_size=0.50,
                                                          				  												random_state=seed)

# Reshaping transcriptomic responses:
y['trn'] = y['trn'].values.reshape([y['trn'].shape[0], 1])
y['dev'] = y['dev'].values.reshape([y['dev'].shape[0], 1])
y['tst'] = y['tst'].values.reshape([y['tst'].shape[0], 1])

# Checking shapes:
y['trn'].shape
X['trn'].shape
y['dev'].shape
X['dev'].shape
y['tst'].shape
X['tst'].shape
X['miss'].shape


#--------------------------------------To run the code on pystan---------------------------------------------#

# Getting the features names prefix:
tmp = X['trn'].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = np.ones(y['trn'].shape) 

# # For subsetting for tests:
# subset1 = np.random.choice(range(X['trn'].shape[0]), size=100)
# subset2 = X['trn'].index[subset1]

# # Storing all the data into a dictionary for pystan:
# df_stan = dict(n_x = X['trn'].loc[subset2,:].shape[0],
#          p_x = X['trn'].shape[1],
#          p_i = np.max(index_x),
#          p_r = X['year'].shape[1],
#          phi = np.max(y['trn'][subset1])*10,
#          index_x = index_x,
#          X = X['trn'].loc[subset2,:],
#          X_r = X['year'].loc[subset2,:],
#          y = y['trn'][subset1].reshape((y['trn'][subset1].shape[0],)))

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['trn'].shape[0],
         p_x = X['trn'].shape[1],
         p_i = np.max(index_x),
         p_r = X['year'].shape[1],
         phi = np.max(y['trn'])*10,
         index_x = index_x,
         X = X['trn'],
         X_r = X['year'],
         y = y['trn'].flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['400'] = model.sampling(data=df_stan, chains=1, iter=400)

# Getting posterior means:
beta_mean = dict()
mu_mean = dict()
beta_mean['400'] = fit['400'].extract()['beta'].mean(axis=0)
mu_mean['400'] = fit['400'].extract()['mu'].mean(axis=0)

# Computing predictions for trn:
y_pred = dict()
y_pred['trn'] = mu_mean['400'] + X['trn'].dot(beta_mean['400'])

# Printing train rMSE errors:
rmse(y['trn'].flatten(), y_pred['trn'])

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev rMSE errors:
rmse(y['dev'].flatten(), y_pred['dev'])

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test rMSE errors:
rmse(y['tst'].flatten(), y_pred['tst'])

# Printing train pearsonr:
pearsonr(y['trn'].flatten(), y_pred['trn'])[0]

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev pearsonr:
pearsonr(y['dev'].flatten(), y_pred['dev'])[0]

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test pearsonr:
pearsonr(y['tst'].flatten(), y_pred['tst'])[0]

# Plots of the observed against the generated:
sns.set_style('whitegrid')
ax = sns.kdeplot(fit['400'].extract()['y_gen'].mean(axis=0), bw=0.5, label='1_400', shade=True)
ax = sns.kdeplot(y['trn'].flatten(), bw=0.5, label='obs', shade=True)
ax.set_title('Observed vs generated data (nchain_niter)')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.clf()

# Plotting:
plt.scatter(y['trn'], y_pred['trn'], color='red')
plt.scatter(y['dev'], y_pred['dev'], color='green')
plt.scatter(y['tst'], y_pred['tst'], color='blue')
# plt.xlim(2.5, 4)
# plt.ylim(2.5, 4)
plt.title('Observed vs predicted data')
plt.xlabel('Observed transcription binned values')
plt.ylabel("Predicted transcription binned values")
plt.show()
