
#------------------------------------------------Modules-----------------------------------------------------#

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Loading libraries:
import pandas as pd
import numpy as np
import os
import pickle
import re
import pystan as ps
import argparse
parser = argparse.ArgumentParser()


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Getting flags:
parser.add_argument("-y", "--y", dest = "y", default = "error", help="Name of the file with the phenotypes")
parser.add_argument("-x", "--x", dest = "x", default = 'error', help="Name of the file with the features")
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model that can be: 'BN' or 'PBN', or 'DBN'")
parser.add_argument("-di", "--dir_in", dest = "dir_in", default = "error", help="Directory of the folder where y and x are stored")
parser.add_argument("-dp", "--dir_proj", dest = "dir_proj", default = "error", help="Directory of the project folder")
parser.add_argument("-do", "--dir_out", dest = "dir_out", default = "error", help="Directory of the folder that will receive the outputs")

args = parser.parse_args()


#---------------------------------------------Loading data---------------------------------------------------#

# # Setting the model:
# model = args.model

# # Directory of the data:
# dir_in = args.dir_in

# # Directory of the project:
# dir_proj = args.dir_proj

# # Directory where outputs will be saved:
# dir_out = args.dir_out

# # Setting directory:
# os.chdir(dir_in)

# # Getting each trait file names:
# if model=='PBN':
# 	y = args.y
# 	x = args.x
# 	y_0 = y.split("&")[0]
# 	y_1 = y.split("&")[1]
# 	x_0 = x.split("&")[0]
# 	x_1 = x.split("&")[1]

# # Loading data:
# if (model=='BN') or bool(re.search('DBN', model)):
# 	# Reading adjusted means:
# 	y = pd.read_csv(args.y, index_col=0)
# 	# Reading feature matrix:
# 	X = pd.read_csv(args.x, index_col=0)

# # Loading data:
# if model=='PBN':
# 	# Reading adjusted means for both traits:
# 	y_0 = pd.read_csv(y_0, index_col=0)
# 	y_1 = pd.read_csv(y_1, index_col=0)
# 	# Reading feature matrix for both traits:
# 	X_0 = pd.read_csv(x_0, index_col=0)
# 	X_1 = pd.read_csv(x_1, index_col=0)


##***** Temp chunck of code:

# Data:

# y = "y_cv1_drymass_k0_trn.csv-y_cv1_height_k0_trn.csv"
# x = "x_cv1_drymass_k0_trn.csv-x_cv1_height_k0_trn.csv"

y = "y_cv1_height_k0_trn.csv"
x = "x_cv1_height_k0_trn.csv"

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model ='GBN-0~1'

# Getting each trait file names:
if model=='PBN':
	y_0 = y.split("-")[0]
	y_1 = y.split("-")[1]
	x_0 = x.split("-")[0]
	x_1 = x.split("-")[1]

# Choose the trait:
trait = 'drymass-height'

# Choose the cv scheme:
cv = 'cv1'

# Directory of the folder where y and x are stored:
dir_in='/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/'

# Directory of the project folder:
dir_proj='/workdir/jp2476/repo/sorghum-multi-trait/'

# Prefix of the output directory:
PREFIX='/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/' + model

# Directory of the folder that will receive the outputs:
dir_out= PREFIX + '/' + cv + '/' + trait 

# Setting directory:
os.chdir(dir_in)

# Loading data:
if (model=='BN') or bool(re.search('DBN', model)) or bool(re.search('GBN', model)):
	# Reading adjusted means:
	y = pd.read_csv(y, index_col=0)
	# Reading feature matrix:
	X = pd.read_csv(x, index_col=0)

# Loading data:
if model=='PBN':
	# Reading adjusted means for both traits:
	y_0 = pd.read_csv(y_0, index_col=0)
	y_1 = pd.read_csv(y_1, index_col=0)
	# Reading feature matrix for both traits:
	X_0 = pd.read_csv(x_0, index_col=0)
	X_1 = pd.read_csv(x_1, index_col=0)


#------------------------------------------Data input for stan-----------------------------------------------#

if model == 'BN':
	# Creating an empty list to receive the dictionaries:
	dict_stan = []
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Droping the first column used just for mapping:
	X = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	for t in group:
		dict_stan.append(dict(p_z = X[index==t].shape[1],
							  n = X[index==t].shape[0],
						 	  Z = X[index==t],
						 	  y = y[index==t].values.flatten(),
						 	  phi = y[index==t].max().values[0]*10)) 

if model == 'PBN':
	# Creating an empty list to receive the dictionaries:
	dict_stan = [] 
	# Subsetting time indexes and groups:
	index1 = X_1.iloc[:,0].values
	index0 = X_0.iloc[:,0].values
	group1 = X_1.iloc[:,0].unique()
	group0 = X_0.iloc[:,0].unique()
	# Droping the first column used just for mapping:
	X_0 = X_0.drop(X_0.columns[0], axis=1)
	X_1 = X_1.drop(X_1.columns[0], axis=1)
	# Case where the first input file have measures over time:
	if len(group1)==1:
		# Building dictionaries:
		for t in group0:
			dict_stan.append(dict(p_z = X_0.shape[1],
								  n_0 = X_0[index0==t].shape[0],
							 	  Z_0 = X_0[index0==t],
							 	  y_0 = y_0[index0==t].values.flatten(),
							 	  n_1 = X_1.shape[0],
							 	  Z_1 = X_1,
							 	  y_1 = y_1.values.flatten(),
							 	  phi = pd.concat([y_0[index0==t], y_1], axis=0).max().values[0]*10))
	# Case where the second input file have measures over time:
	if len(group0)==1:
		# Building dictionaries:
		for t in group1:
			dict_stan.append(dict(p_z = X_0.shape[1],
								  n_0 = X_0.shape[0],
							 	  Z_0 = X_0,
							 	  y_0 = y_0.values.flatten(),
							 	  n_1 = X_1[index1==t].shape[0],
							 	  Z_1 = X_1[index1==t],
							 	  y_1 = y_1[index1==t].values.flatten(),
							 	  phi = pd.concat([y_0, y_1[index1==t]], axis=0).max().values[0]*10))
	# Case where all input files have measures over time:
	if (len(group0)!=1) & (len(group1)!=1):
		# Building dictionaries:
		for t0, t1 in zip(group0, group1):
			dict_stan.append(dict(p_z = X_0.shape[1],
								  n_0 = X_0[index0==t0].shape[0],
							 	  Z_0 = X_0[index0==t0],
							 	  y_0 = y_0[index0==t0].values.flatten(),
							 	  n_1 = X_1[index1==t1].shape[0],
							 	  Z_1 = X_1[index1==t1],
							 	  y_1 = y_1[index1==t1].values.flatten(),
							 	  phi = pd.concat([y_0[index0==t0], y_1[index1==t1]], axis=0).max().values[0]*10))

if model == 'DBN-0~1':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
				 	 phi = y.max().values[0]*10) 

if model == 'DBN-0~2':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
					 n_2 = Z[index==group[2]].shape[0],
				 	 Z_2 = Z[index==group[2]],
				 	 y_2 = y[index==group[2]].values.flatten(),			 	  
				 	 phi = y.max().values[0]*10) 

if model == 'DBN-0~3':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
					 n_2 = Z[index==group[2]].shape[0],
				 	 Z_2 = Z[index==group[2]],
				 	 y_2 = y[index==group[2]].values.flatten(),			 	  
					 n_3 = Z[index==group[3]].shape[0],
				 	 Z_3 = Z[index==group[3]],
				 	 y_3 = y[index==group[3]].values.flatten(),
				 	 phi = y.max().values[0]*10) 

if model == 'DBN-0~4':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
					 n_2 = Z[index==group[2]].shape[0],
				 	 Z_2 = Z[index==group[2]],
				 	 y_2 = y[index==group[2]].values.flatten(),			 	  
					 n_3 = Z[index==group[3]].shape[0],
				 	 Z_3 = Z[index==group[3]],
				 	 y_3 = y[index==group[3]].values.flatten(),
					 n_4 = Z[index==group[4]].shape[0],
				 	 Z_4 = Z[index==group[4]],
				 	 y_4 = y[index==group[4]].values.flatten(),
				 	 phi = y.max().values[0]*10) 

if model == 'DBN-0~5':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
					 n_2 = Z[index==group[2]].shape[0],
				 	 Z_2 = Z[index==group[2]],
				 	 y_2 = y[index==group[2]].values.flatten(),			 	  
					 n_3 = Z[index==group[3]].shape[0],
				 	 Z_3 = Z[index==group[3]],
				 	 y_3 = y[index==group[3]].values.flatten(),
					 n_4 = Z[index==group[4]].shape[0],
				 	 Z_4 = Z[index==group[4]],
				 	 y_4 = y[index==group[4]].values.flatten(),
					 n_5 = Z[index==group[5]].shape[0],
				 	 Z_5 = Z[index==group[5]],
				 	 y_5 = y[index==group[5]].values.flatten(),
				 	 phi = y.max().values[0]*10) 

if model == 'DBN-0~6':
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Subsetting the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)
	# Building dictionaries:
	dict_stan = dict(p_z = Z[index==group[0]].shape[1],
					 p_res = len(group),
					 n_0 = Z[index==group[0]].shape[0],
				 	 Z_0 = Z[index==group[0]],
				 	 y_0 = y[index==group[0]].values.flatten(),
					 n_1 = Z[index==group[1]].shape[0],
				 	 Z_1 = Z[index==group[1]],
				 	 y_1 = y[index==group[1]].values.flatten(),
					 n_2 = Z[index==group[2]].shape[0],
				 	 Z_2 = Z[index==group[2]],
				 	 y_2 = y[index==group[2]].values.flatten(),			 	  
					 n_3 = Z[index==group[3]].shape[0],
				 	 Z_3 = Z[index==group[3]],
				 	 y_3 = y[index==group[3]].values.flatten(),
					 n_4 = Z[index==group[4]].shape[0],
				 	 Z_4 = Z[index==group[4]],
				 	 y_4 = y[index==group[4]].values.flatten(),
					 n_5 = Z[index==group[5]].shape[0],
				 	 Z_5 = Z[index==group[5]],
				 	 y_5 = y[index==group[5]].values.flatten(),
					 n_6 = Z[index==group[6]].shape[0],
				 	 Z_6 = Z[index==group[6]],
				 	 y_6 = y[index==group[6]].values.flatten(),
				 	 phi = y.max().values[0]*10) 

if bool(re.search('GBN', model)):
	# Extract the time upper bound from the model name:
	upper = int(model.split('~')[1])
	# Subset time points:
	dap = X.iloc[:,0].unique()[range(upper+1)]
	# Get time points range:
	time_points = np.linspace(start=0, stop=upper, num=dap.size) + 1	
	# Subset the time covariate and the features:
	Z = X.drop(X.columns[0], axis=1)[X.iloc[:,0].values==dap[0]]
	# Initialize numpy array:
	y_array = np.zeros(shape=(dap.size, Z.shape[0]))
	y_array[:] = np.nan
	# Add phenotypes to two time point column:
	for t in range(dap.size):
		y_array[t, :] = y[X.iloc[:,0].values==dap[t]].values.reshape((Z.shape[0]))
	# Build dictionary:
	dict_stan = dict(n = Z.shape[0],
					 n_t = time_points.size,
					 p_z = Z.shape[1],
					 Z = Z,
					 y = y_array,
					 time_points = time_points,
					 phi = np.max(y_array)*10)


#--------------------------------------Running the Bayesian Network------------------------------------------#

# Setting directory:
os.chdir(dir_proj + "codes")

# For running the Bayesian Network model:s
if model == 'BN':
	# Compiling the Bayesian Network:
	model_stan = ps.StanModel(file='bayesian_network.stan')
	# Creating an empty list:
	fit = []
	# Fitting the model:
	for t in range(len(group)):
		fit.append(model_stan.sampling(data=dict_stan[t], chains=4, iter=400))

# For running the Pleiotropic Bayesian Network model:
if model == 'PBN':
	# Compiling the Pleiotropic Bayesian Network:
	model_stan = ps.StanModel(file='pleiotropic_bayesian_network.stan')
	# Creating an empty list:
	fit = []
	# Fitting the model:
	if len(group1)==1:
		for t in range(len(group0)):
			fit.append(model_stan.sampling(data=dict_stan[t], chains=4, iter=400))
	if len(group0)==1:
		for t in range(len(group1)):
			fit.append(model_stan.sampling(data=dict_stan[t], chains=4, iter=400))
	if (len(group0)!=1) & (len(group1)!=1):
		for t in range(len(group0)):
			fit.append(model_stan.sampling(data=dict_stan[t], chains=4, iter=400))

# Compiling the DBN model:
if model == 'DBN-0~1':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_1.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the DBN model:
if model == 'DBN-0~2':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_2.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the DBN model:
if model == 'DBN-0~3':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_3.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the DBN model:
if model == 'DBN-0~4':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_4.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the DBN model:
if model == 'DBN-0~5':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_5.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the DBN model:
if model == 'DBN-0~6':
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_6.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)

# Compiling the GBN model:
if bool(re.search('GBN', model)):
	model_stan  = ps.StanModel(file='growth_bayesian_network.stan')
	# Fitting the model:
	fit = model_stan.sampling(data=dict_stan, chains=4, iter=400)



fit = model_stan.sampling(data=dict_stan, chains=1, iter=100)

# Fitting the model:
fit = model_stan.sampling(data=dict_stan, chains=4, iter=400, init="random", init_r=0.01)

#---------------------------------Saving outputs from the Bayesian Network-----------------------------------#

# Setting directory:
os.chdir(dir_out)

# Saving stan fit object and model for the BN:
if model == 'BN':
	for t in range(len(group)):
		with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
		    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)

# Saving stan fit object and model for the PBN:
if model == 'PBN':
	if len(group1)==1:
		for t in range(len(group0)):
			with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
			    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)
	if len(group0)==1:
		for t in range(len(group1)):
			with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
			    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)
	if (len(group0)!=1) & (len(group1)!=1):
		for t in range(len(group0)):
			with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
			    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)

# Saving stan fit object and model for the DBN:
if bool(re.search('DBN', model)):
	with open('output_' + model.lower() + '.pkl', 'wb') as f:
	    pickle.dump({'model' : model_stan, 'fit' : fit}, f, protocol=-1)

