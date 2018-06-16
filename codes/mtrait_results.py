
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


#---------------------------------------Reading train and test data------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs was saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Type of cv2 cross-validation schemes:
cv2_type = ['cv2-30~45', 'cv2-30~60', 'cv2-30~75', 'cv2-30~90', 'cv2-30~105']

# Different types of cross-validation splits:
cv_split = ['trn', 'tst']

# Creating a list with the cv1 folds:
cv1_fold = ['k0', 'k1', 'k2', 'k3', 'k4']

# Creating a list with the traits set:
trait_set = ['drymass', 'height']

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Setting the directory:
os.chdir(prefix_out + 'data/cross_validation/')

# Reading cv1 files:
for s in range(len(trait_set)):
	for i in range(len(cv_split)):
		for j in range(len(cv1_fold)):
			# Creating the suffix of the file name for the cv1 data case:
			index = 'cv1_' + trait_set[s] + '_' + cv1_fold[j] + '_' + cv_split[i]
			# Reading data:
			y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
			X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)

# Reading cv2 files:
for t in range(len(cv2_type)):
	for i in range(len(cv_split)):
		for j in range(len(cv1_fold)):
			# Creating the suffix of the file name for the cv2 data case:
			index = cv2_type[t] + '_height_' + cv_split[i]
			# Reading data:
			y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
			X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)


## To do list:
# - Create a tree of directories for PBN, BN (height only), for receiving the cv2 full data scheme for each time point
# - Update the cross-validation code for generating full data cv2 scheme for each data point and feed the new directories 



#----------------------------------------Reading stan-fit outputs--------------------------------------------#

# Creating a list with the models set:
model_set = ['BN', 'PBN', 'DBN']


# Creating a list to receive the stan ouputs:

# Reading ouputs:












