
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

# Setting the model:
model = args.model

# Directory of the data:
dir_in = args.dir_in

# Directory of the project:
dir_proj = args.dir_proj

# Directory where outputs will be saved:
dir_out = args.dir_out

# Setting directory:
os.chdir(dir_in)

# Reading adjusted means:
y = pd.read_csv(args.y, index_col=0)

# Reading feature matrix:
X = pd.read_csv(args.x, index_col=0)

# Setting directory:
os.chdir(dir_proj + "codes")

# Loading external functions:
from external_functions import * 


# ### Temporal chunck of code:

# # Name of the model that can be: 'BN' or 'PBN', or 'DBN':
# model ='PBN'

# # Choose the trait:
# trait = 'height'

# # Choose the cv scheme:
# cv = 'cv1'

# # Directory of the folder where y and x are stored:
# dir_in='/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/'

# # Directory of the project folder:
# dir_proj='/workdir/jp2476/repo/sorghum-multi-trait/'

# # Prefix of the output directory:
# PREFIX='/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/' + model

# # Directory of the folder that will receive the outputs:
# dir_out= PREFIX + '/' + cv + '/' + trait 

# # Setting directory:
# os.chdir(dir_proj + "codes")

# # Loading external functions:
# from external_functions import * 

# # Setting directory:
# os.chdir(dir_in)

# # Reading adjusted means:
# y = pd.read_csv("y_cv1_height_k0_trn.csv", index_col=0)

# # Reading feature matrix:
# X = pd.read_csv("x_cv1_height_k0_trn.csv", index_col=0)


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
os.chdir(dir_proj + "codes")

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
os.chdir(dir_out)

# Saving stan fit object and model:
if model == 'BN':
	for t in range(len(group)):
		with open('output_' + model.lower() + '_fit_' + str(t) + '.pkl', 'wb') as f:
		    pickle.dump({'model' : model_stan, 'fit' : fit[t]}, f, protocol=-1)
