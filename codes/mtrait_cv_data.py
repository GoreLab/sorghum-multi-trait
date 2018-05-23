
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
import re 

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
parser.add_argument("-cv", "--cv", dest = "cv", default = "CV1", help="Cross-validation type")
parser.add_argument("-bt", "--bt", dest = "bt", default = "drymass", help="Biomass type")

args = parser.parse_args()

#------------------------------------------Code parameters---------------------------------------------------#

# Type of cross-validation scheme:
cv = args.cv

# Biomass trait type:
biomass_type = args.bt

# ## Temp:
# cv = "CV1"
# biomass_type = "starch"
# cv = "CV2.30~90"

#--------------------------Building design/feature matrices for height and biomass---------------------------#

# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the data frame with phenotypic data and id's:
df = pd.read_csv("df.csv", header = 0, index_col=0)

# Loading the genomic binned matrix under Cockerham's model:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Changing the class of the year column:
df.year = df.year.astype(object)

# Creating an empty dictionary to receive the feature matrices and response vectors:
X = dict()
y = dict()

# Building the feature matrix for the height:
index = ['loc', 'year', 'dap', 'id_gbs']
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
index = ['loc', 'year', 'id_gbs']
X['biomass'] = pd.get_dummies(df.loc[df.trait=='biomass', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='biomass'])
X['biomass'] = pd.concat([X['biomass'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)

# Removing rows of the missing entries from the feature matrix:
X['biomass'] = X['biomass'][np.invert(df[biomass_type][df.trait=='biomass'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='biomass'
y['biomass'] = df[biomass_type][index][np.invert(df[biomass_type][index].isnull())]


#----------------------------------Preparing data for cross-validation---------------------------------------#

if cv=="CV1":
	# Index for subsetting height data:
	index = df.trait=='height'
	# Index to receive the position of the data frame:
	index_cv = dict()
	# Subsetting data into train and (dev set + test set) for height data:
	X['cv1_height_trn'], X['cv1_height_dev'], y['cv1_height_trn'], y['cv1_height_dev'], index_cv['cv1_height_trn'], index_cv['cv1_height_dev'] = train_test_split(X['height'], 
																																								  y['height'],
										 		                                                																  df.height[index][np.invert(df.height[index].isnull())].index,
										                                                        																  test_size=0.3,
										                                                        																  random_state=1234)
	# Subsetting (dev set + test set) into dev set and test set:
	X['cv1_height_dev'], X['cv1_height_tst'], y['cv1_height_dev'], y['cv1_height_tst'], index_cv['cv1_height_dev'], index_cv['cv1_height_tst'] = train_test_split(X['cv1_height_dev'],
											                                                            		  												  y['cv1_height_dev'],
											                                                            		  												  index_cv['cv1_height_dev'],
										                                                          				  												  test_size=0.50,
										                                                          				  												  random_state=1234)
	# Index for subsetting height data:
	index = df.trait=='biomass'
	# Subsetting data into train and (dev set + test set) for biomass data:
	X['cv1_biomass_trn'], X['cv1_biomass_dev'], y['cv1_biomass_trn'], y['cv1_biomass_dev'], index_cv['cv1_biomass_trn'], index_cv['cv1_biomass_dev'] = train_test_split(X['biomass'], 
																																								        y['biomass'],
										 		                                                																        df[biomass_type][index][np.invert(df[biomass_type][index].isnull())].index,
										                                                        																        test_size=0.3,
										                                                        																        random_state=1234)
	# Subsetting (dev set + test set) into dev set and test set:
	X['cv1_biomass_dev'], X['cv1_biomass_tst'], y['cv1_biomass_dev'], y['cv1_biomass_tst'], index_cv['cv1_biomass_dev'], index_cv['cv1_biomass_tst'] = train_test_split(X['cv1_biomass_dev'],
											                                                            		  												        y['cv1_biomass_dev'],
											                                                            		  												        index_cv['cv1_biomass_dev'],
										                                                          				  												        test_size=0.50,
										                                                          				  												        random_state=1234)
	# Reshaping responses:
	y['cv1_height_trn'] = np.reshape(y['cv1_height_trn'], (y['cv1_height_trn'].shape[0], 1))
	y['cv1_height_dev'] = np.reshape(y['cv1_height_dev'], (y['cv1_height_dev'].shape[0], 1))
	y['cv1_height_tst'] = np.reshape(y['cv1_height_tst'], (y['cv1_height_tst'].shape[0], 1))
	y['cv1_biomass_trn'] = np.reshape(y['cv1_biomass_trn'], (y['cv1_biomass_trn'].shape[0], 1))
	y['cv1_biomass_dev'] = np.reshape(y['cv1_biomass_dev'], (y['cv1_biomass_dev'].shape[0], 1))
	y['cv1_biomass_tst'] = np.reshape(y['cv1_biomass_tst'], (y['cv1_biomass_tst'].shape[0], 1))
	# Checking shapes of the matrices related to height:
	X['cv1_height_trn'].shape
	y['cv1_height_trn'].shape
	X['cv1_height_dev'].shape
	y['cv1_height_dev'].shape
	X['cv1_height_tst'].shape
	y['cv1_height_tst'].shape
	# Checking shapes of the matrices related to biomass:
	X['cv1_biomass_trn'].shape
	y['cv1_biomass_trn'].shape
	X['cv1_biomass_dev'].shape
	y['cv1_biomass_dev'].shape
	X['cv1_biomass_tst'].shape
	y['cv1_biomass_tst'].shape
	# Setting directory to save cv data:
	os.chdir(prefix_out + 'data/cross_validation/cv1')
	# Saving height data:
	X['cv1_height_trn'].to_csv('x_cv1_height_trn.csv')
	pd.DataFrame(y['cv1_height_trn'], index=index_cv['cv1_height_trn']).to_csv('y_cv1_height_trn.csv')
	X['cv1_height_dev'].to_csv('x_cv1_height_dev.csv')
	pd.DataFrame(y['cv1_height_dev'], index=index_cv['cv1_height_dev']).to_csv('y_cv1_height_dev.csv')
	X['cv1_height_tst'].to_csv('x_cv1_height_tst.csv')
	pd.DataFrame(y['cv1_height_tst'], index=index_cv['cv1_height_tst']).to_csv('y_cv1_height_tst.csv')
	# Saving biomass data:
	X['cv1_biomass_trn'].to_csv('x_cv1_biomass_' + biomass_type + '_trn.csv')
	pd.DataFrame(y['cv1_biomass_trn'], index=index_cv['cv1_biomass_trn']).to_csv('y_cv1_biomass_' + biomass_type + '_trn.csv')
	X['cv1_biomass_dev'].to_csv('x_cv1_biomass_' + biomass_type + '_dev.csv')
	pd.DataFrame(y['cv1_biomass_dev'], index=index_cv['cv1_biomass_dev']).to_csv('y_cv1_biomass_' + biomass_type + '_dev.csv')
	X['cv1_biomass_tst'].to_csv('x_cv1_biomass_' + biomass_type + '_tst.csv')
	pd.DataFrame(y['cv1_biomass_tst'], index=index_cv['cv1_biomass_tst']).to_csv('y_cv1_biomass_' + biomass_type + '_tst.csv')

if bool(re.search('CV2', cv)):
	# Varible to define trait:
	trait = 'height'
	# Getting the upper interval limit of the days after planting (dap) that will compose the training set:
	upper = float(cv.split('~')[1])
	# Name of the current training set:
	tmp= 'cv2.' + str(int(df.dap.min())) + '~' + str(int(upper)) + '_' + trait + '_'
	cv_type = [tmp + s for s in ['trn', 'dev', 'tst']]
	# Subsetting the training set:
	index = np.invert(df.height[df.trait==trait].isnull()) & (df.dap <= upper)
	X[cv_type[0]] = X[trait].loc[df.height[index].index]
	y[cv_type[0]] = y[trait][df.height[index].index]
	# Subsetting the non training set:
	index = np.invert(df.height[df.trait==trait].isnull()) & (df.dap > upper)
	X[cv_type[1]] = X[trait].loc[df.height[index].index]
	y[cv_type[1]] =  y[trait][df.height[index].index]
	# Splitting non training set into dev and test sets:
	X[cv_type[1]], X[cv_type[2]], y[cv_type[1]], y[cv_type[2]] = train_test_split(X[cv_type[1]],
											                                      y[cv_type[1]],
										                                          test_size=0.50,
										                                          random_state=1234)
	# Setting directory to save cv data:
	os.chdir(prefix_out + 'data/cross_validation/' + cv.lower())
	X[cv_type[0]].to_csv('x_' + cv.lower() + '_' + trait + '_trn.csv')
	pd.DataFrame(y[cv_type[0]]).to_csv('y_' + cv.lower() + '_' + trait + '_trn.csv')
	X[cv_type[1]].to_csv('x_' + cv.lower() + '_' + trait + '_dev.csv')
	pd.DataFrame(y[cv_type[1]]).to_csv('y_' + cv.lower() + '_' + trait + '_dev.csv')
	X[cv_type[2]].to_csv('x_' + cv.lower() + '_' + trait + '_tst.csv')
	pd.DataFrame(y[cv_type[2]]).to_csv('y_' + cv.lower() + '_' + trait + '_tst.csv')
