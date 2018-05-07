
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

#-c core -a number of alternatives
parser.add_argument("-c", "--core", dest = "core", default = 0, help="Current core where the analysis is happing", type=int)
parser.add_argument("-nalt", "--nalternatives", dest = "nalt", default = 1, help="Number of alternative models per bin", type=int)

args = parser.parse_args()

print( "cores {} alt {}".format(
        args.core,
        args.nalt,
     ))

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


#------------------------------------------Code parameters---------------------------------------------------#

# Specifying the model
model = 'BN'				 # 'DBN' or 'BN' or 'BNP'

# Choosing the data structure to be analysed:
structure = 'biomass_trn'

# # Current core where the analysis is happening:
# core = args.core

# # Number of alternative runs per bin:
# n_alt = args.nalt

# ## Temp:
# core=0
# n_alt=2

# Seed to recover the analysis:
seed = int(str(core) + str(n_alt))

#----------------------------------------Bayesian Network code-----------------------------------------------#

# Getting the features names prefix:
tmp = X[structure].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = pd.get_dummies(df.year.loc[X[structure].index]) 


#-----------------------------------------Deep learning code-------------------------------------------------#

# Checking shapes of the matrices related to height:
X[structure] = X[structure].transpose()
y[structure] = y[structure].transpose()

# Checking shapes of the matrices related to height:
X[structure].shape
y[structure].shape

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
                              max=5,                   # Maximum number of hidden units
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
batch_size_lst = sample_batch(n = X[structure].shape[1],      # Number of observations, or examples, or target
                              n_guess = n_alt,            # Number of guesses
                              same_str = X[structure].shape[1])           # False: Random guess; [Value] insert a value to replicate

# Sampling dropout keep prob hyperparameter:
np.random.seed(seed)
keep_prob_lst = sample_interval(min = 0.0001,      # Minimum of the quantitative interval
                                max = 1,           # Maximum of the quantitative interval
                                n_guess = n_alt,   # Number of guesses
                                same_str = False)      # False: Random guess; [Value] insert a value to replicate


## Creating folders on the project to store the alternative results:

# Getting in the output directory:
bash_line1 = "cd " + prefix_out + "outputs/" + model + "/" + structure + "/;"

# Creating folders to store the results:
bash_line2 = "for i in $(seq 0 " + str(n_alt-1)+ "); do mkdir core"+ str(core) + "_alt${i}" + "; done;"

# Running proces on unix shell:
subprocess.call(bash_line1 + bash_line2, shell=True)

# For choosing either GPU:
if proc_type == "GPU":
    proc = '/device:GPU:0'

# or CPU:
if proc_type == "CPU":
	proc = '/device:CPU:0'

## Training optimization step:
for alt in range(n_alt):
     with tf.device(proc):
      # Start hyperparameters:
      batch_mode = batch_mode_lst[alt]
      dropout_mode = dropout_mode_lst[alt]
      starter_learning_rate = starter_learning_rate_lst[alt]            # Starter learning rate
      n_layers = n_layers_lst[alt]
      h_units = h_units_lst[alt]                                        # Number of hidden units
      batch_size = batch_size_lst[alt]                                  # Batch size
      total_batch = int(len(X[structure].transpose()) / batch_size)       # Total batch size
      # Creating variables for batch normalization:
      const = tf.constant(0.01)
      # Creating symbolic variables (Tensors Planceholders):
      A0 = tf.placeholder(tf.float32, shape=(X[structure].shape[0], batch_size))  # Symbolic input training variable
      Y = tf.placeholder(tf.float32, shape=(1, batch_size))                     # Symbolic response training variable
      Lamb = tf.placeholder(tf.float32)                                         # Symbolic regularization parameter
      keep_prob = tf.placeholder(tf.float32)                                    # Symbolic probability (to remove hidden units) for dropout
      # Initializing variables:
      W = []                                    # Symbolic weights
      B = []                                    # Symbolic bias
      Z = []                                    # Linear activation
      A = []                                    # Output of the non-linear or linear activation
      fro_norm=[]
      # Initializing dropout variables:
      if dropout_mode == True:
        D = []
      # Initializing batch norm variables:
      if batch_mode == True:
        Z_mean = [None] * n_layers                                            # Mean of the linear activation
        Z_var = [None] * n_layers                                             # Variance of the linear activation
        Z_tilde = [None] * n_layers                                           # Normalized output of the linear activation
        beta = []                                                             # Batch norm parameter
        gamma = []                                                            # Batch norm parameter
        beta.append(tf.Variable(tf.zeros([h_units[0], 1]), name="beta1"))
        gamma.append(tf.Variable(tf.ones([h_units[0], 1]), name="gamma1"))
      # First hidden layer:
      W.append(tf.Variable(tf.random_uniform([h_units[0], X[structure].shape[0]])*const,
                        dtype=tf.float32, name="W1"))
      B.append(tf.Variable(tf.zeros([h_units[0], 1]),
                        dtype=tf.float32, name="B1"))
      Z.append(tf.matmul(W[0], A0) + B[0])
      A.append(tf.nn.relu(Z[0]))
      # Batch norm for the first hidden layer:
      if batch_mode==True:
        Z_mean[0], Z_var[0] = tf.nn.moments(x=Z[0], axes=[1], keep_dims=True)
        Z_tilde[0] = tf.nn.batch_normalization(x = Z[0],
                                          mean = Z_mean[0],
                                          variance = Z_var[0],
                                          offset = beta[0],
                                          scale = gamma[0],
                                          variance_epsilon = epsilon) 
      # Dropout for the first hidden layer:
      if dropout_mode==True:
        D.append(tf.less(tf.random_uniform(shape=Z[0].get_shape()), keep_prob))
        D[0].set_shape(Z[0].get_shape())
        A[0] = tf.divide(tf.multiply(A[0], tf.cast(D[0], tf.float32)), keep_prob) 
        A[0] = tf.reshape(A[0], Z[0].get_shape())
      # Adding other hidden layers:
      for i in range(1, n_layers):
        W.append(tf.Variable(tf.random_uniform([h_units[i], h_units[i-1]])*const,
                          dtype=tf.float32, name=("W" + str(i+1))))
        B.append(tf.Variable(tf.zeros([h_units[i], 1]),
                          dtype=tf.float32, name=("B" + str(i+1))))
        Z.append(tf.matmul(W[i], A[i-1]) + B[i])
        A.append(tf.nn.relu(Z[i]))
        # Batch norm:
        if batch_mode==True:
          beta.append(tf.Variable(tf.zeros([h_units[i], 1]), name=("beta" + str(i+1))))
          gamma.append(tf.Variable(tf.ones([h_units[i], 1]), name=("gamma" + str(i+1))))
          Z_mean[i], Z_var[i] = tf.nn.moments(x=Z[i], axes=[1], keep_dims=True)
          Z_tilde[i] = tf.nn.batch_normalization(x = Z[i],
                                            mean = Z_mean[i],
                                            variance = Z_var[i],
                                            offset = beta[i],
                                            scale = gamma[i],
                                            variance_epsilon = epsilon)
        # Dropout:
        if dropout_mode==True:
          D.append(tf.less(tf.random_uniform(shape=Z[i].get_shape()), keep_prob))
          D[i].set_shape(Z[i].get_shape())
          A[i] = tf.divide(tf.multiply(A[i], tf.cast(D[i], tf.float32)), keep_prob) 
          A[i] = tf.reshape(A[i], Z[i].get_shape())
      # Parameters of the output layer:
      W.append(tf.Variable(tf.random_uniform([h_units[n_layers], h_units[n_layers-1]])*const,
                        dtype=tf.float32, name="W_out"))
      B.append(tf.Variable(tf.zeros([h_units[n_layers], 1]),
                        dtype=tf.float32, name="B_out"))
      # Linear activation:
      Z.append(tf.matmul(W[n_layers], A[n_layers-1]) + B[n_layers])
      # Output ReLu activation:
      Y_out = tf.nn.relu(Z[n_layers])
      # Frobenius norm:
      for i in range(0, n_layers+1):
        fro_norm.append(tf.norm(tensor=W[i], ord='fro', axis=[-2, -1]))
      # Summing values:
      fro_norm = tf.reduce_sum(fro_norm, keepdims=True)
      # Regularization:
      const2 = tf.multiply(tf.constant(2, dtype=tf.float32), tf.constant(X[structure].shape[0], dtype=tf.float32)) 
      reg = tf.multiply(tf.divide(Lamb, const2), fro_norm) 
      # Exponential decay:
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=0.5, decay_steps=len(setIter)*total_batch, name="learning_rate")
      # Cost function:
      cost = tf.add(tf.losses.mean_squared_error(labels=Y,
                                                 predictions=Y_out), reg)
      # For storing the cost values over time:
      tf.summary.scalar("costs", tf.squeeze(cost))
      # Define optimizer:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
      if proc_type == "GPU":
           session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
      else:
           server = tf.train.Server.create_local_server()
           session = tf.Session(server.target, config=tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1))
      # Initialize session:
      init = tf.global_variables_initializer()
      session.run(init)
      # Merge all the summaries and write them out to:
      merged_summary = tf.summary.merge_all()
      writer = tf.summary.FileWriter(prefix_out + "outputs/" + model + "/" + structure + "/core" + str(core) + "_alt" + str(alt) + "/")
      writer.add_graph(session.graph)
      # Optimizing the Deep Neural Network (DNN):
      for epoch in range_epoch:
       if epoch > 0:
            print("\n", "Cost:", "\n", out[0]);
       X_batch = np.array_split(X[structure].transpose(), total_batch)
       Y_batch = np.array_split(y[structure].transpose(), total_batch)
       for j in setIter:
           for i in range(total_batch):
               if X_batch[i].shape[0] > batch_size:
                   rows_ = range(X_batch[i].shape[0] - batch_size)
                   x_batch, y_batch = np.delete(X_batch[i],rows_,0).transpose(), np.delete(Y_batch[i], rows_,0).transpose()
               else: x_batch, y_batch = X_batch[i].transpose(), Y_batch[i].transpose()
               s, _, out = session.run([merged_summary, optimizer, cost], feed_dict={A0:x_batch, Y:y_batch, Lamb:lamb_lst[alt], keep_prob:keep_prob_lst[alt]})
               writer.add_summary(s, i)
      results[alt] = out
      saver = tf.train.Saver()
      os.chdir(prefix_out + "outputs/" + model + "/" + structure + "/core" + str(core) + "_alt" + str(alt))       
      save_path = "./core" + str(core) + "_alt" + str(alt)
      saver.save(session, save_path)
      tf.reset_default_graph()  
































