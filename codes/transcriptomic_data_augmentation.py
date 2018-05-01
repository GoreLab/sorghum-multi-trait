
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
T_bin = pd.read_csv("T_bin.csv", header = 0, index_col=0)

#----------------------------------Preparing data for cross-validation---------------------------------------#

## Temp:
r=0
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
index_wbin = np.array_split(T_bin.columns, n_cores)

# for r in range(index_wbin[core].size):


# Creating an empty dictionary to receive feature matrices, and responses:
X = dict()
y = dict()

# Indexing the phenotype (transcription at bin r)
y['full'] = T_bin[index_wbin[core].values[r]]

# Feature matrix considering only individuals with genomic and transcriptomic data:
X['full'] = W_bin.loc[T_bin.index]

# Indexing the genomic data of the real missing transcriptomic dat:
X['miss'] = W_bin.loc[np.invert(W_bin.index.isin(T_bin.index))]
 
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

# Transposing matrices and vectors:
y['trn'] = y['trn'].transpose()
X['trn'] = X['trn'].transpose()
y['dev'] = y['dev'].transpose()
X['dev'] = X['dev'].transpose()
y['tst'] = y['tst'].transpose()
X['tst'] = X['tst'].transpose()
X['miss'] = X['miss'].transpose()

# Checking shapes:
y['trn'].shape
X['trn'].shape
y['dev'].shape
X['dev'].shape
y['tst'].shape
X['tst'].shape
X['miss'].shape

#-----------------------------------------Going deeply: Training----------------------------------------------#

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
batch_mode = False

# Dropout (True or False):
dropout_mode = False  

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
batch_size_lst = sample_batch(n = X['trn'].shape[1],      # Number of observations, or examples, or target
                              n_guess = n_alt,            # Number of guesses
                              same_str = X['trn'].shape[1])           # False: Random guess; [Value] insert a value to replicate

# Sampling dropout keep prob hyperparameter:
np.random.seed(seed)
keep_prob_lst = sample_interval(min = 0.0001,      # Minimum of the quantitative interval
                                max = 1,           # Maximum of the quantitative interval
                                n_guess = n_alt,   # Number of guesses
                                same_str = 1)      # False: Random guess; [Value] insert a value to replicate


## Creating folders on the project to store the alternative results:

# Getting in the output directory:
bash_line1 = "cd " + prefix_out + "outputs/rnaseq_imp;"

# Creating folders to store the results:
bash_line2 = "for i in $(seq 0 " + str(n_alt-1)+ "); do mkdir core"+ str(core) + "_alt${i}_bin" + str(r) + "; done;"

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
      starter_learning_rate = starter_learning_rate_lst[alt]            # Starter learning rate
      n_layers = n_layers_lst[alt]
      h_units = h_units_lst[alt]                                        # Number of hidden units
      batch_size = batch_size_lst[alt]                                  # Batch size
      total_batch = int(len(X['trn'].transpose()) / batch_size)       # Total batch size
      # Creating variables for batch normalization:
      const = tf.constant(0.01)
      # Creating symbolic variables (Tensors Planceholders):
      A0 = tf.placeholder(tf.float32, shape=(X['trn'].shape[0], batch_size))  # Symbolic input training variable
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
      W.append(tf.Variable(tf.random_uniform([h_units[0], X['trn'].shape[0]])*const,
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
      const2 = tf.multiply(tf.constant(2, dtype=tf.float32), tf.constant(X['trn'].shape[0], dtype=tf.float32)) 
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
           session = tf.Session(server.target)
      # Initialize session:
      init = tf.global_variables_initializer()
      session.run(init)
      # Merge all the summaries and write them out to:
      merged_summary = tf.summary.merge_all()
      writer = tf.summary.FileWriter(prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + "/")
      writer.add_graph(session.graph)
      # Optimizing the Deep Neural Network (DNN):
      for epoch in range_epoch:
       if epoch > 0:
            print("\n", "Cost:", "\n", out[0]);
       X_batch = np.array_split(X['trn'].transpose(), total_batch)
       Y_batch = np.array_split(y['trn'].transpose(), total_batch)
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
      os.chdir(prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r))       
      save_path = "./core" + str(core) + "_alt" + str(alt) + "_bin" + str(r)
      saver.save(session, save_path)
      tf.reset_default_graph()  


#----------------------------------------Going deeply: Developing--------------------------------------------#


# To reset the previous graph:
tf.reset_default_graph()

# Number of simulations:
n_sim=1

# Small epsilon value for batch norm
epsilon = 1e-7

# Number of data sets:
n_sets = 4

# Initializing lists:
Y_pred_lst_sets = []
Y_pred_lst = np.empty(shape=[n_alt, X['trn'].shape[1] , n_sim])

# Foward pass:
for c in range(n_sets):
  # For training set:
  if c==0:
    X_tmp = X['trn']
  # For development set:
  if c==1:
    X_tmp = X['dev']
  # For testing set:
  if c==2:
    X_tmp = X['tst']
  # For missing set:
  if c==3:
    X_tmp = X['miss']
  # Initializing list:
  Y_pred_lst = np.empty(shape=[n_alt, X_tmp.shape[1] , n_sim])
  for alt in range(n_alt):
    # Importing results:
    tf.reset_default_graph()
    n_layers = n_layers_lst[alt]
    session = tf.Session() 
    save_path = prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + "/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + ".meta"
    saver = tf.train.import_meta_graph(save_path, clear_devices=True)
    save_path = prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r)
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
      Y_pred_lst[alt, :, sim] = np.dot(session.run("W_out:0"), A[n_layers-1]) + session.run("B_out:0")   # Initialize linear predictor
  # Adding to the sets list:
  Y_pred_lst_sets.append(Y_pred_lst)


# Updating the number of sets:
n_sets=3

# Initialize a variable to receive metrics:
rmse_sets = np.empty([n_alt, n_sets])
r2_sets = np.empty([n_alt, n_sets])
mic_sets = np.empty([n_alt, n_sets])

# Getting metrics:
for c in range(n_sets):
  # Getting final predictions:
  Y_pred = np.mean(Y_pred_lst_sets[c], axis=2)
  # Computing RMSE of the prediction using the dev set:
  if c==0:
      y_tmp = y['trn']
  elif c==1:
      y_tmp = y['dev']
  else:
      y_tmp = y['tst']
  for alt in range(n_alt):
      rmse_sets[alt, c] = rmse(y_tmp, Y_pred[alt,:])
      r2_sets[alt, c] = r2_score(np.squeeze(y_tmp), Y_pred[alt,:])
      mic_sets[alt, c] = normalized_mutual_info_score(np.squeeze(y_tmp), Y_pred[alt,:])


# Printing RMSE:
print(np.round(np.sort(rmse_sets[:,0], axis=0),4))
print(np.round(np.sort(rmse_sets[:,1], axis=0),4))
print(np.round(np.sort(rmse_sets[:,2], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,0], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,1], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,2], axis=0),4))

# Printing r2:
print(np.round(np.sort(r2_sets[:,0], axis=0)[::-1],4))
print(np.round(np.sort(r2_sets[:,1], axis=0)[::-1],4))
print(np.round(np.sort(r2_sets[:,2], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,0], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,1], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,2], axis=0)[::-1],4))

# Printing Mutual Information Criteria:
print(np.round(np.sort(mic_sets[:,0], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,1], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,2], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,0], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,1], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,2], axis=0)[::-1],4))

#---------------------------------------Storing the best prediction------------------------------------------#

# Setting directory:
os.chdir(prefix_out + "outputs/rnaseq_imp/predicted_bins")

# Getting the index of the best predictions:
index = np.argsort(rmse_sets[:,1], axis=0)

# Getting the mean across dropout vectors:
Y_pred = np.mean(Y_pred_lst_sets[3], axis=2)

# Create empty submission dataframe
out = pd.DataFrame()

# Adding the predictions to the output file:
out[index_wbin[core].values[r]] = Y_pred[index[0], :]

# Insert ID and Predictions into dataframe
out.index = X['miss'].columns.values

# Output submission file
out.to_csv(index_wbin[core].values[r] + "_predicted.csv")

