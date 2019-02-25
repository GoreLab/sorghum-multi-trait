

#------------------------------------------------Modules-----------------------------------------------------#

# # Import tensorflow to use GPUs on keras:
import tensorflow as tf

# Set keras with GPUs
import keras
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12}, intra_op_parallelism_threads = 12, inter_op_parallelism_threads = 12) 
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12}) 
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




#----------------------------------------------Development----------------------------------------------------file, -#

# Prefix of the project:
PREFIX = ''

# Reading the data:
df = dict()


















