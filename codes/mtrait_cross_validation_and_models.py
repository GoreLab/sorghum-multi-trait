
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
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


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

# Building the feature matrix for the height:
index = ['loc', 'year', 'dap']
X_height = pd.get_dummies(df.loc[df.trait=='height', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='height'])
X_height = np.hstack((np.dot(tmp, W_bin.loc[tmp.columns.tolist()]), X_height))

# Removing rows of the missing entries from the feature matrix:
X_height = X_height[np.invert(df.height[df.trait=='height'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='height'
y_height = df.height[index][np.invert(df.height[index].isnull())]

# Building the feature matrix for the biomass:
index = ['loc', 'year', 'adf', 'moisture', 'ndf', 'protein', 'starch']
X_biomass = pd.get_dummies(df.loc[df.trait=='biomass', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='biomass'])
X_biomass = np.hstack((np.dot(tmp, W_bin.loc[tmp.columns.tolist()]), X_biomass))

# Removing rows of the missing entries from the feature matrix:
X_biomass = X_biomass[np.invert(df.drymass[df.trait=='biomass'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='biomass'
y_biomass = df.drymass[index][np.invert(df.drymass[index].isnull())]


#-----------------------------------Preparing data for cross-validation--------------------------------------#




