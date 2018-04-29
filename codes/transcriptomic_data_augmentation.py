
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

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
# prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
# prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


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

# Creating an empty dictionary to receive feature matrices, and responses:
X = dict()
y = dict()

# Indexing the phenotype (transcription at bin j)
y['full'] = T_bin['bin_1']

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
                                                        									random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X['dev'], X['tst'], y['dev'], y['tst'], index_cv['dev'], index_cv['tst'] = train_test_split(X['dev'],
	                                                            		  					y['dev'],
	                                                            		  					index_cv['dev'],
                                                          				  					test_size=0.50,
                                                          				  					random_state=1234)

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
















