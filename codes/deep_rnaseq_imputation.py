

#------------------------------------------------Modules-----------------------------------------------------#

# # Import tensorflow to use GPUs on keras:
import tensorflow as tf

# Set keras with GPUs
import keras
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12}, intra_op_parallelism_threads = 12, inter_op_parallelism_threads = 12) 
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12}) 
sess = tf.Session(config=config)

# keras.backend.set_session(sess)
# keras.backend.tensorflow_backend._get_available_gpus()

# Import keras tools:
from keras.models import Sequential, Model, Input
from keras.layers import MaxPooling1D, AveragePooling1D, Conv1D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint

# Import libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import ceil
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import pickle
from scipy.stats import skew
from scipy.stats import moment
import matplotlib.pyplot as plt
import seaborn as sns

import gc

#------------------------------------------------Function----------------------------------------------------file, -#

# Print some features of the data:
def get_info(data):
    for i in data.columns:
        print('feature, min, max: {}, {}, {}'.format(i, data[i].min(), data[i].max()))

# Split data into different sets:
def split_cv(input, train_size, seed, only_target):
    if only_target==False:
        # Split data again into train and a subset:
        trn_y, dev_y, trn_x, dev_x = train_test_split(input[0], input[1], test_size=(1-train_size), random_state=seed)
        # Split the subset into dev and test:
        dev_y, tst_y, dev_x, tst_x = train_test_split(dev_y, dev_x, test_size=0.5, random_state=seed)
        # Return data frames:
        return(trn_y, dev_y, tst_y, trn_x, dev_x, tst_x)
    if only_target==True:
        # Split data again into train and a subset:
        trn_y, dev_y = train_test_split(input[0], test_size=(1-train_size), random_state=seed)
        # Split the subset into dev and test:
        dev_y, tst_y = train_test_split(dev_y, test_size=0.5, random_state=seed)
        gc.collect()
        # Return data frames:
        return(trn_y, dev_y, tst_y)

def generate_cv(y, x, train_size, n_pad, seed):
    # Split data into train, dev and test sets:
    y['trn'], y['dev'], y['tst'], x['trn'], x['dev'], x['tst'] = split_cv([y['obs'], x['obs']],
                                                                          seed = seed,
                                                                          train_size = train_size,
                                                                          only_target = False)
    # Padding feature matrices:
    if n_pad != 0:
        for key in list(x.keys()):
            x[key] = pad_col(x[key], n_pad)
    gc.collect()
    # print('shape, set: {}, {}'.format(x[key].shape, key))
    return y, x

# Multiperceptron neural networtk:
def MNN(input_shape):
	input = Input(input_shape)
	net = Dense(3, activation="relu")(input) 
	net = Dropout(0.7)(net) 
	net = BatchNormalization()(net)
	net = Dense(3, activation="relu")(net) 
	net = Dropout(0.7)(net) 
	net = BatchNormalization()(net)
	net = Dense(2, activation="relu")(net) 
	net = Dropout(0.7)(net) 
	net = BatchNormalization()(net)
	net = Dense(2, activation="relu")(net)
	net = Dropout(0.7)(net) 
	net = BatchNormalization()(net)
	net = Dense(1, activation="relu")(net) 
	model = Model(inputs=input,outputs=net)
	gc.collect()
	return model

# Function to create a residual block:
def ResBlock1D(x,filters,kernel_size,pool=False):
    res = x
    if pool:
        x = MaxPooling1D(pool_size=2)(x)
        res = Conv1D(filters=filters, kernel_size=1, activation= 'linear', strides=2, padding="same")(res)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Dropout(0.7)(res)
    out = BatchNormalization()(x)
    out = Conv1D(filters=filters, kernel_size=kernel_size, activation= 'linear', strides=1, padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.7)(out)
    out = Conv1D(filters=filters, kernel_size=kernel_size, activation= 'linear', strides=1, padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.7)(out)
    out = keras.layers.add([res,out])
    gc.collect()
    return out

# Create the model architecture:
def CNN(input_shape):
    input_ = Input(input_shape)
    net = Conv1D(filters=16, kernel_size=128, activation= 'relu', strides=1, padding="same")(input_)
    net = ResBlock1D(net,filters=16, kernel_size=64)
    net = ResBlock1D(net,filters=32, kernel_size=32, pool=True)
    net = ResBlock1D(net,filters=64, kernel_size=16, pool=True)
    net = Flatten()(net)
    net = Dense(8, activation= 'linear')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.7)(net)
    net = Dense(4, activation= 'linear')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.7)(net)
    net = Dense(1, activation="linear")(net) 
    model = Model(inputs=input_,outputs=net)
    gc.collect()
    return model

# To train the model:
def train_model(y_train, x_train, y_dev, x_dev, lr, batch_size, epochs, type):
    # Define the model:
    if type == 'MNN':
    	model = MNN(input_shape = (x_train.shape[1],))
    if type == 'CNN':
    	model = CNN(input_shape = (x_train.shape[1], 1))
    # Compile the model:
    model.compile(optimizer=Adam(lr=lr, clipnorm=1.), loss="mse")
    # Fit the model:
    model.fit(x=x_train,
              y=y_train,
              validation_data=[x_dev, y_dev],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    gc.collect()
    # Return the model:
    return model


#----------------------------------------------Development----------------------------------------------------file, -#

# Prefix of the project:
PROJ = '~/Documents/sorghum-multi-trait/'

# Reading the data:
df = dict()
df['dna'] = pd.read_csv(PROJ + 'data/W_bin.csv', index_col=0)
df['rna'] = pd.read_csv(PROJ + 'data/T.csv', index_col=0)

# Get info from the transcriptomic data:
get_info(df['rna'])

# Heatmap of the bins:
sns.heatmap(df['dna'].corr(), cmap='PiYG')
plt.show()

# Initialize dictionaries to receive targets and feature matrix:
y = dict()
x = dict()

# Subset target vector:
y['obs'] = df['rna'].iloc[:,0]

# Subset features from the common entries:
index =  df['dna'].index.isin(df['rna'].index)
x['obs'] = df['dna'][index]
x['obs'] = x['obs'].reindex(df['rna'].index)

# Generate train and test set:
y, x = generate_cv(y=y, x=x, train_size=0.7, n_pad=0, seed=1234)

# Train te model
hist = train_model(y_train = y['trn'],
 				   x_train = x['trn'],
 				   y_dev = y['dev'],
 				   x_dev = x['dev'],
 				   lr = 0.001,
 				   batch_size = 256,
 				   epochs = 10000,
 				   type = 'MNN')

# Predictions:
y_pred = dict()
y_pred['trn'] = hist.predict(x['trn'])
y_pred['dev'] = hist.predict(x['dev'])
y_pred['tst'] = hist.predict(x['tst'])

# Print accuracies:
pearsonr(np.array(y['trn'], y_pred['trn'])[0][0]

# Reshape feature matrices:
x['trn'] = x['trn'].as_matrix().reshape(x['trn'].shape[0], x['trn'].shape[1], 1)
x['dev'] = x['dev'].as_matrix().reshape(x['dev'].shape[0], x['dev'].shape[1], 1)
x['tst'] = x['tst'].as_matrix().reshape(x['tst'].shape[0], x['tst'].shape[1], 1)

# Train te model
hist = train_model(y_train = y['trn'],
 				   x_train = x['trn'],
 				   y_dev = y['dev'],
 				   x_dev = x['dev'],
 				   lr = 0.01,
 				   batch_size = 256,
 				   epochs = 1000,
 				   type = 'CNN')

# Predictions:
y_pred = dict()
y_pred['trn'] = hist.predict(x['trn'])
y_pred['dev'] = hist.predict(x['dev'])
y_pred['tst'] = hist.predict(x['tst'])

# Print accuracies:
pearsonr(y['trn'], y_pred['trn'].reshape(y_pred['trn'].size))[0]
pearsonr(y['dev'], y_pred['dev'].reshape(y_pred['dev'].size))[0]
pearsonr(y['tst'], y_pred['tst'].reshape(y_pred['tst'].size))[0]































