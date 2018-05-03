
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

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


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


#--------------------------------------Preparing data for pystan---------------------------------------------#

# Getting the features names prefix:
tmp = X['biomass_trn'].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = pd.get_dummies(df.year.loc[X['biomass_trn'].index]) 

# For subsetting for tests:
subset1 = np.random.choice(range(X['biomass_trn'].shape[0]), size=100)
subset2 = X['biomass_trn'].index[subset1]

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['biomass_trn'].loc[subset2,:].shape[0],
			   p_x = X['biomass_trn'].shape[1],
			   p_i = np.max(index_x),
			   p_r = X['year'].shape[1],
			   phi = np.max(y['biomass_trn'][subset1])*10,
			   index_x = index_x,
			   X = X['biomass_trn'].loc[subset2,:],
			   X_r = X['year'].loc[subset2,:],
			   y = y['biomass_trn'][subset1].reshape((y['biomass_trn'][subset1].shape[0],)))

#--------------------------------------To run the code on pystan---------------------------------------------#

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['300'] = model.sampling(data=df_stan, chains=1, iter=300)
fit['400'] = model.sampling(data=df_stan, chains=1, iter=400)
fit['600'] = model.sampling(data=df_stan, chains=1, iter=600)
fit['1_2000'] = model.sampling(data=df_stan, chains=1, iter=2000)
fit['4_2000'] = model.sampling(data=df_stan, chains=4, iter=2000)

# Getting posterior means:
beta_mean = dict()
mu_mean = dict()
beta_mean['300'] = fit['300'].extract()['beta'].mean(axis=0)
mu_mean['300'] = fit['300'].extract()['mu'].mean(axis=0)
beta_mean['400'] = fit['400'].extract()['beta'].mean(axis=0)
mu_mean['400'] = fit['400'].extract()['mu'].mean(axis=0)
beta_mean['600'] = fit['600'].extract()['beta'].mean(axis=0)
mu_mean['600'] = fit['600'].extract()['mu'].mean(axis=0)
beta_mean['1_2000'] = fit['1_2000'].extract()['beta'].mean(axis=0)
mu_mean['1_2000'] = fit['1_2000'].extract()['mu'].mean(axis=0)
beta_mean['4_2000'] = fit['4_2000'].extract()['beta'].mean(axis=0)
mu_mean['4_2000'] = fit['4_2000'].extract()['mu'].mean(axis=0)

# Computing predictions for trn:
y_pred = dict()
y_pred['trn_300'] = mu_mean['300'] + X['biomass_trn'].loc[subset2,:].dot(beta_mean['300'])
y_pred['trn_400'] = mu_mean['400'] + X['biomass_trn'].loc[subset2,:].dot(beta_mean['400'])
y_pred['trn_600'] = mu_mean['600'] + X['biomass_trn'].loc[subset2,:].dot(beta_mean['600'])
y_pred['trn_1_2000'] = mu_mean['1_2000'] + X['biomass_trn'].loc[subset2,:].dot(beta_mean['1_2000'])
y_pred['trn_4_2000'] = mu_mean['4_2000'] + X['biomass_trn'].loc[subset2,:].dot(beta_mean['4_2000'])

# Computing predictions for dev:
y_pred['dev_300'] = mu_mean['300'] + X['biomass_dev'].dot(beta_mean['300'])
y_pred['dev_400'] = mu_mean['400'] + X['biomass_dev'].dot(beta_mean['400'])
y_pred['dev_600'] = mu_mean['600'] + X['biomass_dev'].dot(beta_mean['600'])
y_pred['dev_1_2000'] = mu_mean['1_2000'] + X['biomass_dev'].dot(beta_mean['1_2000'])
y_pred['dev_4_2000'] = mu_mean['4_2000'] + X['biomass_dev'].dot(beta_mean['4_2000'])

# Computing predictions for test:
y_pred['tst_300'] = mu_mean['300'] + X['biomass_tst'].dot(beta_mean['300'])
y_pred['tst_400'] = mu_mean['400'] + X['biomass_tst'].dot(beta_mean['400'])
y_pred['tst_600'] = mu_mean['600'] + X['biomass_tst'].dot(beta_mean['600'])
y_pred['tst_1_2000'] = mu_mean['1_2000'] + X['biomass_tst'].dot(beta_mean['1_2000'])
y_pred['tst_4_2000'] = mu_mean['4_2000'] + X['biomass_tst'].dot(beta_mean['4_2000'])


# Printing train rMSE errors:
round(rmse(y['biomass_trn'][subset1].flatten(), y_pred['trn_300']),4)
round(rmse(y['biomass_trn'][subset1].flatten(), y_pred['trn_400']),4)
round(rmse(y['biomass_trn'][subset1].flatten(), y_pred['trn_600']),4)
round(rmse(y['biomass_trn'][subset1].flatten(), y_pred['trn_1_2000']),4)
round(rmse(y['biomass_trn'][subset1].flatten(), y_pred['trn_4_2000']),4)

# Printing dev rMSE errors:
round(rmse(y['biomass_dev'].flatten(), y_pred['dev_300']),4)
round(rmse(y['biomass_dev'].flatten(), y_pred['dev_400']),4)
round(rmse(y['biomass_dev'].flatten(), y_pred['dev_600']),4)
round(rmse(y['biomass_dev'].flatten(), y_pred['dev_1_2000']),4)
round(rmse(y['biomass_dev'].flatten(), y_pred['dev_4_2000']),4)

# Printing test rMSE errors:
round(rmse(y['biomass_tst'].flatten(), y_pred['tst_300']),4)
round(rmse(y['biomass_tst'].flatten(), y_pred['tst_400']),4)
round(rmse(y['biomass_tst'].flatten(), y_pred['tst_600']),4)
round(rmse(y['biomass_tst'].flatten(), y_pred['tst_1_2000']),4)
round(rmse(y['biomass_tst'].flatten(), y_pred['tst_4_2000']),4)

# Setting directory:
os.chdir(prefix_out + "plots")

# Plots of the observed against the generated:
sns.set_style('whitegrid')
ax = sns.kdeplot(fit['300'].extract()['y_gen'].mean(axis=0), bw=0.5, label='1_300', shade=True)
ax = sns.kdeplot(fit['400'].extract()['y_gen'].mean(axis=0), bw=0.5, label='1_400', shade=True)
ax = sns.kdeplot(fit['600'].extract()['y_gen'].mean(axis=0), bw=0.5, label='1_600', shade=True)
ax = sns.kdeplot(fit['1_2000'].extract()['y_gen'].mean(axis=0), bw=0.5, label='1_2000', shade=True)
ax = sns.kdeplot(fit['4_2000'].extract()['y_gen'].mean(axis=0), bw=0.5, label='4_2000', shade=True)
ax = sns.kdeplot(y['biomass_trn'][subset1].flatten(), bw=0.5, label='obs', shade=True)
ax.set_title('Observed vs generated data (nchain_niter)')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.savefig(prefix_out + 'plots/' + 'benchmark_niter_biomass_density_obs_gen' + '.pdf')
plt.show()
plt.clf()

# Plotting:
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev_300'], label="300", alpha=0.3)
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev_400'], label="400", alpha=0.3)
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev_600'], label="600", alpha=0.3)
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev_1_2000'], label="1_2000", alpha=0.3)
plt.scatter(y['biomass_dev'].flatten(), y_pred['dev_4_2000'], label="4_2000", alpha=0.3)
# plt.xlim(2.5, 4)
# plt.ylim(2.5, 4)
plt.legend()
plt.title('Observed vs predicted data from the dev set (nchain_niter)')
plt.xlabel('Observed data')
plt.ylabel("Predicted data")
# plt.savefig(prefix_out + 'plots/' + 'benchmark_niter_biomass_density_obs_pred' + '.pdf')
plt.show()
plt.clf()
