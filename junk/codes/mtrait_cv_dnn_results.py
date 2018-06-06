
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
# matplotlib.use('GTK') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import os

import tensorflow as tf
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
parser.add_argument("-nc", "--ncore", dest = "ncore", default = 0, help="Current core where the analysis is happing", type=int)
parser.add_argument("-nalt", "--nalt", dest = "nalt", default = 1, help="Number of alternatives", type=int)
parser.add_argument("-d", "--data", dest = "data", default = "cv1_biomass_trn", help="Data set to be analysed")
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")

args = parser.parse_args()

#------------------------------------------Code parameters---------------------------------------------------#

# # Current core where the analysis is happening:
# n_core = args.ncore

# # Number of alternative runs per bin:
# n_alt = args.nalt

# # Choosing the data structure to be analysed:
# struc = args.data

# # Specifying the model
# model = args.model         # 'DBN' or 'BN' or 'BNP'

# # Type of cross-validation scheme:
# cv = args.cv

## Temp:
n_core = 40
n_alt = 10
struc = "cv1_height"
model = "DNN"
cv = "CV1"

#--------------------------------------------Reading data----------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + 'data/cross_validation/' + cv.lower())

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Loading the data:
X['trn'] = pd.read_csv('x_' + struc + '_trn.csv', header = 0, index_col=0)
y['trn'] = pd.read_csv('y_' + struc + '_trn.csv', header = 0, index_col=0)
X['dev'] = pd.read_csv('x_' + struc + '_dev.csv', header = 0, index_col=0)
y['dev'] = pd.read_csv('y_' + struc + '_dev.csv', header = 0, index_col=0)
X['tst'] = pd.read_csv('x_' + struc + '_tst.csv', header = 0, index_col=0)
y['tst'] = pd.read_csv('y_' + struc + '_tst.csv', header = 0, index_col=0)

if struc=="cv1_biomass":
  # Subsetting just the desired factors:
  index = X['trn'].columns.str.contains('|'.join(['loc','year', 'bin']))
  X['trn'] = X['trn'].loc[:,index]
  index = X['dev'].columns.str.contains('|'.join(['loc','year', 'bin']))
  X['dev'] = X['dev'].loc[:,index]
  index = X['tst'].columns.str.contains('|'.join(['loc','year', 'bin']))
  X['tst'] = X['tst'].loc[:,index]

if struc=="cv1_height":
  # Subsetting just the desired factors:
  index = X['trn'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
  X['trn'] = X['trn'].loc[:,index]
  index = X['dev'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
  X['dev'] = X['dev'].loc[:,index]
  index = X['tst'].columns.str.contains('|'.join(['loc','year', 'dap', 'bin']))
  X['tst'] = X['tst'].loc[:,index]

# Transposing matrices:
X['trn'] = X['trn'].transpose()
y['trn'] = y['trn'].transpose()
X['dev'] = X['dev'].transpose()
y['dev'] = y['dev'].transpose()
X['tst'] = X['tst'].transpose()
y['tst'] = y['tst'].transpose()

# Checking shapes:
X['trn'].shape
y['trn'].shape
X['dev'].shape
y['dev'].shape
X['tst'].shape
y['tst'].shape

#---------------------------------------------Foward pass----------------------------------------------------#

# Number of simulations for dropout:
n_sim = 100

# Number of data sets:
n_sets = 3

# Initializing lists:
Y_pred_lst_sets = []

# Foward pass:
for c in range(n_sets):
	for m in range(n_core):
		# Seed to recover the analysis:
		seed = int(str(m) + str(n_alt))
		# A list to receive the results:
		results = [None] * n_alt
		# Generating iterations:
		setIter = range(1000)
		# Generating epochs:
		range_epoch = range(50)
		# Small epsilon value for batch norm
		epsilon = 1e-7
		# Type of processor (CPU or GPU):
		proc_type = "CPU"
		# Number of hidden layers:
		np.random.seed(seed)
		n_layers_lst = sample_n_h_layers(min=1,          # Maximum number of hidden units
		                                 max=5,          # Number of hidden layers
		                                 n_guess=n_alt,  # Number of guesses
		                                 same_str=2)     # False: Random guess; [Some architecture]: architecture to be replicated across guesses
		# Batch norm (True or False):
		np.random.seed(seed)
		batch_mode_lst = np.random.choice([True, False], size=n_alt)
		# Dropout (True or False):
		np.random.seed(seed)
		dropout_mode_lst = np.random.choice([True, False], size=n_alt) 
		# Sampling the hidden units:
		np.random.seed(seed)
		h_units_lst =  sample_h_units(min=1,                    # Minimum number of hidden units
		                              max=5,                    # Maximum number of hidden units
		                              n_layers=n_layers_lst,    # Number of hidden layers (it should be a list)
		                              n_guess=n_alt,            # Number of alternatives or guesses:
		                              same_str=False)           # False: Random guess; [Some architecture]: architecture to be replicated across guesses
		# Sampling the initial learning rate:
		np.random.seed(seed)
		starter_learning_rate_lst = sample_interval(min = 0.0001,        # Minimum of the quantitative interval
		                                            max = 1,             # Maximum of the quantitative interval
		                                            n_guess = n_alt,     # Number of guesses
		                                            same_str = False)    # False: Random guess; [Value] insert a value to replicate
		# Sampling Frobenius regularizer:
		np.random.seed(seed)
		lamb_lst = sample_interval(min = 0.0001,    # Minimum of the quantitative interval
		               max = 2,                     # Maximum of the quantitative interval
		               n_guess = n_alt,             # Number of guesses
		               same_str = False)            # False: Random guess; [Value] insert a value to replicate
		# Sampling batch size:
		np.random.seed(seed)
		batch_size_lst = sample_batch(n = X['trn'].shape[1],      # Number of observations, or examples, or target
		                              n_guess = n_alt,            # Number of guesses
		                              same_str = X['trn'].shape[1])           # False: Random guess; [Value] insert a value to replicate
		# Sampling dropout keep prob hyperparameter:
		np.random.seed(seed)
		keep_prob_lst = sample_interval(min = 0.0001,      # Minimum of the quantitative interval
		                                max = 1,           # Maximum of the quantitative interval
		                                n_guess = n_alt,   # Number of guesses
		                                same_str = False)      # False: Random guess; [Value] insert a value to replicate
		# For training set:
		if c==0:
		  X_tmp = X['trn']
		# For development set:
		if c==1:
		  X_tmp = X['dev']
		# For testing set:
		if c==2:
		  X_tmp = X['tst']
		# For hidden set:
		if m==0:
		  # Initializing list:
		  Y_pred_lst = np.empty(shape=[n_alt, X_tmp.shape[1] , n_sim, n_core])
		for alt in range(n_alt):
		  # Importing results:
		  tf.reset_default_graph()
		  n_layers = n_layers_lst[alt]
		  batch_mode = batch_mode_lst[alt]
		  dropout_mode = dropout_mode_lst[alt]
		  session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) 
		  save_path = prefix_out + "outputs/cross_validation/" + model.lower() + "/" + struc + "/core" + str(m) + "_alt" + str(alt) + "/core" + str(m) + "_alt" + str(alt) + ".meta"
		  saver = tf.train.import_meta_graph(save_path, clear_devices=True)
		  save_path = prefix_out + "outputs/cross_validation/" + model.lower() + "/" + struc + "/core" + str(m) + "_alt" + str(alt)
		  saver.restore(session,tf.train.latest_checkpoint(save_path))
		  for sim in range(n_sim):
		    for l in range(n_layers):
		      # Initializing variables:
		      if l==0:
		        Z = []
		        A = []          
		      # Initializing batch norm variables:
		      if batch_mode==True and l==0:
		        batch_mean = []
		        batch_var = []
		        Z_norm = []
		        Z_tilde = []
		      # Initializing dropout variables:n_sets
		      if dropout_mode==True and l==0:
		        D = []
		      # Linear activation:
		      if l==0:
		        Z.append(np.dot(session.run("W1:0"), X_tmp) + session.run("B1:0"))            
		      if l!=0:
		        Z.append(np.dot(session.run("W{}:0".format(l+1)), A[l-1]) + session.run("B{}:0".format(l+1)))   
		      # Batch norm:
		      if batch_mode==True:
		        batch_mean.append(moment(Z[l], moment=1, axis=1))                                                         # Getting the mean across examples
		        batch_var.append(moment(Z[l], moment=2, axis=1))                                                          # Getting the variance across examples
		        batch_mean[l] = batch_mean[l].reshape([Z[l].shape[0], 1])                                                 # Reshaping moments
		        batch_var[l] = batch_var[l].reshape([Z[l].shape[0],1])                                                    # Reshaping moments    
		        Z_norm.append((Z[l] - batch_mean[l]) / np.sqrt(batch_var[l] + epsilon))                                     # Normalizing output of the linear combination
		        Z_tilde.append(session.run("gamma{}:0".format(l+1)) * Z_norm[l] + session.run("beta{}:0".format(l+1)))   # Batch normalization
		        A.append(np.maximum(Z_tilde[l], 0))                                                                 # Relu activation function
		      else:
		        A.append(np.maximum(Z[l], 0))
		      # Dropout:
		      if dropout_mode==True:
		        np.random.seed((alt+sim))                                                      # Fixing the seed
		        D.append(np.random.rand(A[l].shape[0], A[l].shape[1]) < keep_prob_lst[alt])   # Generating random binary indicators
		        A[l] = np.divide(np.multiply(A[l], D[l]), keep_prob_lst[alt])              # Dropout regularization
		    # Output layer:
		    Y_pred_lst[alt, :, sim, m] = np.dot(session.run("W_out:0"), A[n_layers-1]) + session.run("B_out:0")   # Initialize linear predictor
		    # Reset the computational graph:
		    tf.reset_default_graph()
	# Adding to the sets list:
	Y_pred_lst_sets.append(Y_pred_lst)


#---------------------------------------------Development----------------------------------------------------#

# Initialize a variable to receive metrics:
rmse_sets = np.empty([n_alt, n_core, n_sets])
rmse_sets[:] = np.nan
cor_sets = np.empty([n_alt, n_core, n_sets])
cor_sets[:] = np.nan
r2_sets = np.empty([n_alt, n_core, n_sets])
r2_sets[:] = np.nan

# Getting metrics:
for c in range(n_sets):
  # Getting final predictions:
  Y_pred = np.mean(Y_pred_lst_sets[c], axis=2)
  # Computing RMSE of the prediction using the dev set:
  for m in range(n_core):
    # Getting data set:
    if c==0:
        y_tmp = y['trn'].values.flatten()
    elif c==1:
        y_tmp = y['dev'].values.flatten()
    else:
        y_tmp = y['tst'].values.flatten()
    for alt in range(n_alt):
    	# If it is lacking result:
    	if (np.any(np.isnan(Y_pred[alt,:,m]))):
    		print("Core {} and alternative {} failed to run on {}".format(m, alt, ['trn', 'dev', 'tst'][c]))
    	# If it is not lacking result:
    	if (np.invert(np.any(np.isnan(Y_pred[alt,:,m])))):
    		rmse_sets[alt, m, c] = rmse(y_tmp, Y_pred[alt,:,m])
    		cor_sets[alt, m, c] = pearsonr(y_tmp, Y_pred[alt,:,m])[0]
    		r2_sets[alt, m, c] = r2_score(y_tmp, Y_pred[alt,:,m])


# Getting the index of the best predictions:
index = get_index(array=rmse_sets[:,:,1], n=1)

# Printing RMSE:
print(np.round(rmse_sets[index[0], index[1],:], 4))

# Printing Pearson correlation:
print(np.round(cor_sets[index[0], index[1],:], 4))

# Printing r2:
print(np.round(r2_sets[index[0], index[1],:], 4))

# Type of sets:
y_pred = dict()
tmp = ['trn', 'dev', 'tst']

# Storing the best prediction:
for c in range(n_sets):
  # Getting final predictions:
  y_pred[tmp[c]] = np.mean(Y_pred_lst_sets[c], axis=2)[index[0],:,index[1]]

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
