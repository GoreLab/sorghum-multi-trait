
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

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


#---------------------------------------------Loading data---------------------------------------------------#

# Setting the model:
model = 'BN'

# Name of the output file:
name_out = 'bn'

# Setting directory:
os.chdir(prefix_out + "data/cross_validation")

# Reading adjusted means:
y = pd.read_csv("y_cv1_height_k0_trn.csv", index_col=0)

# Reading feature matrix:
X = pd.read_csv("x_cv1_height_k0_trn.csv", index_col=0)


#------------------------------------------Data input for stan-----------------------------------------------#

if model == 'BN':
	# Creating an empty list to receive the dictionaries:
	dict_stan = [] 
	# Subsetting time indexes and groups:
	index = X.iloc[:,0].values
	group = X.iloc[:,0].unique()
	# Building dictionaries:
	for t in group:
		dict_stan.append(dict(n = X[index==t].shape[0],
						 	  p_z = X[index==t].shape[1],
						 	  Z = X[index==t],
						 	  y = y[index==t].values.flatten(),
						 	  phi = np.max(y[index==t]).values[0]*10)) 

# To do list:
# - Change the code to run serial analysis for each DAP measure for PBN
# - Prepare data input for stan for all models
# - Prepare the outputs directories

#--------------------------------------Running the Bayesian Network------------------------------------------#

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the BN model:
if model == 'BN':
	model_stan = ps.StanModel(file='bayesian_network.stan')
	# Creating an empty list:
	fit = []
	# Fitting the model:
	for t in range(len(group)):
		fit.append(model_stan.sampling(data=dict_stan[t], chains=4, iter=400))

# Compiling the PBN model:
if model == 'PBN':
	model_stan = ps.StanModel(file='pleiotropic_bayesian_network.stan')

# Compiling the DBN model:
if bool(re.search('DBN', model)):
	model_stan = ps.StanModel(file='dynamic_bayesian_network_0_6.stan')


#---------------------------------Saving outputs from the Bayesian Network-----------------------------------#

# Setting directory:
os.chdir(prefix_out + 'outputs/cross_validation/' + model.lower())

# Saving stan fit object and model:
if model == 'BN':
	for t in range(len(group)):
		with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
		    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)
