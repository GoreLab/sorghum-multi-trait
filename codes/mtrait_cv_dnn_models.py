
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

# Getting flags:
parser.add_argument("-c", "--core", dest = "core", default = 0, help="Current core where the analysis is happing", type=int)
parser.add_argument("-nalt", "--nalt", dest = "nalt", default = 1, help="Number of alternatives", type=int)
parser.add_argument("-d", "--data", dest = "data", default = "cv1_biomass_trn", help="Data set to be analysed")
parser.add_argument("-m", "--model", dest = "model", default = "BN", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")

args = parser.parse_args()


#------------------------------------------Code parameters---------------------------------------------------#

# Current core where the analysis is happening:
core = args.core

# Number of alternative runs per bin:
n_alt = args.nalt

# Choosing the data structure to be analysed:
structure = args.data

# Specifying the model
model = args.model         # 'DBN' or 'BN' or 'BNP'

# Type of cross-validation scheme:
cv = args.cv

# ## Temp:
# core=0
# n_alt=2
# structure="cv1_biomass"
# model="DNN"
# cv = "CV1"

# Seed to recover the analysis:
seed = int(str(core) + str(n_alt))

#--------------------------------------------Reading data----------------------------------------------------#

# Setting the directory:
os.chdir(prefix_out + 'data/cross_validation/' + cv.lower())

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Loading the data:
X['trn'] = pd.read_csv('x_' + structure + '_trn.csv', header = 0, index_col=0)
y['trn'] = pd.read_csv('y_' + structure + '_trn.csv', header = 0, index_col=0)
X['dev'] = pd.read_csv('x_' + structure + '_dev.csv', header = 0, index_col=0)
y['dev'] = pd.read_csv('y_' + structure + '_dev.csv', header = 0, index_col=0)
X['tst'] = pd.read_csv('x_' + structure + '_tst.csv', header = 0, index_col=0)
y['tst'] = pd.read_csv('y_' + structure + '_tst.csv', header = 0, index_col=0)

# Subsetting just the desired factors:
index = X['trn'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['trn'] = X['trn'].loc[:,index]
index = X['dev'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['dev'] = X['dev'].loc[:,index]
index = X['tst'].columns.str.contains('|'.join(['loc','year', 'bin']))
X['tst'] = X['tst'].loc[:,index]

#-----------------------------------------Deep learning code-------------------------------------------------#

# Checking shapes of the matrices related to height:
X['trn'] = X['trn'].transpose()
y['trn'] = y['trn'].transpose()

# Checking shapes of the matrices related to height:
X['trn'].shape
y['trn'].shape

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


# Creating folders on the project to store the alternative results:

# Getting in the output directory:
bash_line1 = "cd " + prefix_out + "outputs/cross_validation/" + model.lower() + "/" + structure + "/;"

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
      Y_out = Z[n_layers]
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
           session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1))
      # Initialize session:
      init = tf.global_variables_initializer()
      session.run(init)
      # Merge all the summaries and write them out to:
      merged_summary = tf.summary.merge_all()
      writer = tf.summary.FileWriter(prefix_out + "outputs/cross_validation/" + model.lower() + "/" + structure + "/core" + str(core) + "_alt" + str(alt) + "/")
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
      os.chdir(prefix_out + "outputs/cross_validation/" + model.lower() + "/" + structure + "/core" + str(core) + "_alt" + str(alt))       
      save_path = "./core" + str(core) + "_alt" + str(alt)
      saver.save(session, save_path)
      tf.reset_default_graph()  
