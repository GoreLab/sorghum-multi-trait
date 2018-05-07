
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
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")

args = parser.parse_args()

#--------------------------Building design/feature matrices for height and biomass---------------------------#

# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the data frame with phenotypic data and id's:
df = pd.read_csv("df.csv", header = 0, index_col=0)

# Loading the genomic binned matrix under Cockerham's model:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Loading the transcriptomic binned matrix:
T_bin = pd.read_csv("T_bin.csv", header = 0, index_col=0)

# Changing the class of the year column:
df.year = df.year.astype(object)

# Creating an empty dictionary to receive the feature matrices and response vectors:
X = dict()
y = dict()

# Building the feature matrix for the height:
index = ['loc', 'year', 'dap']
X['height'] = pd.get_dummies(df.loc[df.trait=='height', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='height'])
X['height'] = pd.concat([X['height'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)

# Removing rows of the missing entries from the feature matrix:
X['height'] = X['height'][np.invert(df.height[df.trait=='height'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='height'
y['height'] = df.height[index][np.invert(df.height[index].isnull())]

# Building the feature matrix for the biomass:
index = ['loc', 'year']
X['biomass'] = pd.get_dummies(df.loc[df.trait=='biomass', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='biomass'])
X['biomass'] = pd.concat([X['biomass'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)

# Removing rows of the missing entries from the feature matrix:
X['biomass'] = X['biomass'][np.invert(df.drymass[df.trait=='biomass'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='biomass'
y['biomass'] = df.drymass[index][np.invert(df.drymass[index].isnull())]

#------------------------------------------Code parameters---------------------------------------------------#

# # Current core where the analysis is happening:
# core = args.core

# # Specifying the model
# model = args.model         # 'DBN' or 'BN' or 'BNP'

# # Type of cross-validation scheme:
# cv = args.cv

# Choosing the data structure to be analysed:
structure = 'biomass_trn'

## Temp:
core = 0
model = "BN"
cv = "CV1"

# Seed to recover the analysis:
seed = core

#----------------------------------Preparing data for cross-validation---------------------------------------#

# Index for subsetting height data:
index = df.trait=='height'

# Index to receive the position of the data frame:
index_cv = dict()

# Subsetting data into train and (dev set + test set) for height data:
X['height_trn'], X['height_dev'], y['height_trn'], y['height_dev'], index_cv['height_trn'], index_cv['height_dev'] = train_test_split(X['height'], 
																														  y['height'],
 		                                                																  df.height[index][np.invert(df.height[index].isnull())].index,
                                                        																  test_size=0.3,
                                                        																  random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X['height_dev'], X['height_tst'], y['height_dev'], y['height_tst'], index_cv['height_dev'], index_cv['height_tst'] = train_test_split(X['height_dev'],
	                                                            		  												  y['height_dev'],
	                                                            		  												  index_cv['height_dev'],
                                                          				  												  test_size=0.50,
                                                          				  												  random_state=1234)

# Index for subsetting height data:
index = df.trait=='biomass'

# Subsetting data into train and (dev set + test set) for biomass data:
X['biomass_trn'], X['biomass_dev'], y['biomass_trn'], y['biomass_dev'], index_cv['biomass_trn'], index_cv['biomass_dev'] = train_test_split(X['biomass'], 
																														        y['biomass'],
 		                                                																        df.drymass[index][np.invert(df.drymass[index].isnull())].index,
                                                        																        test_size=0.3,
                                                        																        random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X['biomass_dev'], X['biomass_tst'], y['biomass_dev'], y['biomass_tst'], index_cv['biomass_dev'], index_cv['biomass_tst'] = train_test_split(X['biomass_dev'],
	                                                            		  												        y['biomass_dev'],
	                                                            		  												        index_cv['biomass_dev'],
                                                          				  												        test_size=0.50,
                                                          				  												        random_state=1234)

# Reshaping responses:
y['height_trn'] = np.reshape(y['height_trn'], (y['height_trn'].shape[0], 1))
y['height_dev'] = np.reshape(y['height_dev'], (y['height_dev'].shape[0], 1))
y['height_tst'] = np.reshape(y['height_tst'], (y['height_tst'].shape[0], 1))
y['biomass_trn'] = np.reshape(y['biomass_trn'], (y['biomass_trn'].shape[0], 1))
y['biomass_dev'] = np.reshape(y['biomass_dev'], (y['biomass_dev'].shape[0], 1))
y['biomass_tst'] = np.reshape(y['biomass_tst'], (y['biomass_tst'].shape[0], 1))

# Checking shapes of the matrices related to height:
X['height_trn'].shape
y['height_trn'].shape
X['height_dev'].shape
y['height_dev'].shape
X['height_tst'].shape
y['height_tst'].shape

# Checking shapes of the matrices related to biomass:
X['biomass_trn'].shape
y['biomass_trn'].shape
X['biomass_dev'].shape
y['biomass_dev'].shape
X['biomass_tst'].shape
y['biomass_tst'].shape

#----------------------------Subdivision of the height data into mini-batches--------------------------------#

# Subsetting the full set of names of the inbred lines phenotyped for biomass:
index_mbatch = df.id_gbs[df.trait=='height'].drop_duplicates()

# Size of the mini-batch
size_mbatch = 4

# Splitting the list of names of the inbred lines into 4 sublists for indexing the mini-batches:
index_mbatch = np.array_split(index_mbatch, size_mbatch)

# Type of sets:
tmp = ['trn', 'dev', 'tst']

# Indexing the mini-batches for the height trait:
for k in tmp:
	for i in range(size_mbatch):
		# Getting the positions on the height training set related to the mini-batch i:
		index = df.id_gbs.loc[index_cv['height_' + k]].isin(index_mbatch[i])
		# Indexing height values of the mini-batch i:
		X['height_'+ k + '_mb_' + str(i)] = X['height_' + k][index]
		y['height_'+ k + '_mb_' + str(i)] = y['height_' + k][index]
		index_cv['height_'+ k +'_mb_' + str(i)] = index_cv['height_' + k][index]
		# Printing shapes:
		X['height_'+ k + '_mb_' + str(i)].shape
		y['height_'+ k + '_mb_' + str(i)].shape

#----------------------------------------Bayesian Network code-----------------------------------------------#

# Getting the features names prefix:
tmp = X[structure].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = pd.get_dummies(df.year.loc[X[structure].index]) 

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X[structure].shape[0],
         p_x = X[structure].shape[1],
         p_i = np.max(index_x),
         p_r = X['year'].shape[1],
         phi = np.max(y[structure])*10,
         index_x = index_x,
         X = X[structure],
         X_r = X['year'],
         y = y[structure].flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit[structure] = model.sampling(data=df_stan, chains=1, iter=400)

#-----------------------------------------Saving stan outputs------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + "data")

# Saving stan fit object and model:
# with open("model_fit.pkl", "wb") as f:
#     pickle.dump({'model' : model, 'fit' : fit}, f, protocol=-1)


# Loading stan fit object and model:
with open("model_fit.pkl", "rb") as f:
    data_dict = pickle.load(f)

# Indexing the fit object and model
model = data_dict['model']
fit = data_dict['fit']

# Extracting the outputs:
outs = fit['biomass_trn'].extract()

# Getting the predictions:
y_pred = dict()
y_pred['trn'] = outs['mu'].mean(axis=0) + X['biomass_trn'].dot(outs['beta'].mean(axis=0))
y_pred['dev'] = outs['mu'].mean(axis=0) + X['biomass_dev'].dot(outs['beta'].mean(axis=0))
y_pred['tst'] = outs['mu'].mean(axis=0) + X['biomass_tst'].dot(outs['beta'].mean(axis=0))

# Density plots of different data types:
sns.set_style('whitegrid')
ax = sns.kdeplot(y['biomass_trn'].flatten(), bw=0.5, label='train set', shade=True)
ax = sns.kdeplot(y['biomass_dev'].flatten(), bw=0.5, label='dev set', shade=True)
ax = sns.kdeplot(outs['y_gen'].mean(axis=0), bw=0.5, label='gen set', shade=True)
ax = sns.kdeplot(y_pred['dev'], bw=0.5, label='pred set', shade=True)
ax.set_title('Density of different data types')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.clf()

# Scatter plots of different data types:
tmp = dict()
tmp['trn'] = np.polyfit(y['biomass_trn'].flatten(), y_pred['trn'], 1)
tmp['dev'] = np.polyfit(y['biomass_dev'].flatten(), y_pred['dev'], 1)
tmp['tst'] = np.polyfit(y['biomass_tst'].flatten(), y_pred['tst'], 1)
plt.scatter(y['biomass_trn'].flatten(), y_pred['trn'], label="trn", alpha=0.3)
plt.plot(y['biomass_trn'].flatten(), tmp['trn'][0] * y['biomass_trn'].flatten() + tmp['trn'][1])
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev'], label="dev", alpha=0.3)
plt.plot(y['biomass_dev'].flatten(), tmp['dev'][0] * y['biomass_dev'].flatten() + tmp['dev'][1])
plt.scatter(y['biomass_tst'].flatten(), y_pred['tst'], label="tst", alpha=0.3)
plt.plot(y['biomass_tst'].flatten(), tmp['tst'][0] * y['biomass_tst'].flatten() + tmp['tst'][1])
plt.legend()
plt.title('Scatter pattern of different data types')
plt.xlabel('Observed data')
plt.ylabel("Predicted data")
plt.xlim(0, 26)
plt.ylim(0, 26)
plt.show()
plt.clf()

# Printing rMSE:
rmse(y['biomass_trn'].flatten(), y_pred['trn'])
rmse(y['biomass_dev'].flatten(), y_pred['dev'])
rmse(y['biomass_tst'].flatten(), y_pred['tst'])

# Printing pearsonr:
pearsonr(y['biomass_trn'].flatten(), y_pred['trn'])[0]
pearsonr(y['biomass_dev'].flatten(), y_pred['dev'])[0]
pearsonr(y['biomass_tst'].flatten(), y_pred['tst'])[0]

# Printing r2:
r2_score(y['biomass_trn'].flatten(), y_pred['trn'])
r2_score(y['biomass_dev'].flatten(), y_pred['dev'])
r2_score(y['biomass_tst'].flatten(), y_pred['tst'])

