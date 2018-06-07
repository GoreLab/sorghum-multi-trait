
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns

import os

from sklearn.model_selection import KFold

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


#-----------------------------------Loading Adjusted Means and plotting--------------------------------------#

# Setting directory:
os.chdir(prefix_out + "outputs/first_step_analysis")

# Readomg adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Changing class of the dap:
df.dap = df.dap.fillna(0).astype(int)

# Setting directory:
os.chdir(prefix_out + "data")

# Reading marker binned matrix:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Filtering just rows of the marker matrix which we have phenotypes:
W_bin = W_bin.loc[df.id_gbs.unique()]

#--------------------------Splitting data into groups for 5th-fold cross-validation--------------------------#

# Number of folds:
n_fold = 5

# Creating five folds:
kf = KFold(n_splits=n_fold, shuffle=True, random_state=1234)

# Getting the splits:
index = kf.split(df['id_gbs'].drop_duplicates())

# Initializing lists to receive the random indexes:
trn_index = []
tst_index = []

# Getting the indexes:
for trn, tst in index:
	trn_index.append(df['id_gbs'].drop_duplicates().iloc[trn])
	tst_index.append(df['id_gbs'].drop_duplicates().iloc[tst])

# Creating dictionary with the data from the first cross-validation scheme:
y = dict()
X = dict()

# Building the feature matrix using all data set entries:
X_all = pd.get_dummies(df['id_gbs']) 
X_all = X_all.dot(W_bin.loc[X_all.columns])
X_all = pd.concat([df.dap, X_all], axis=1)

# Creating different sets:
sets = ['trn', 'tst']

# Building the sets of the data for the CV1 scheme:
for s in sets:
	for t in df.trait.unique():
		for i in range(n_fold):
			if s == 'trn':
				# Logical vector for indexation:
				index = df.id_gbs.isin(trn_index[i]) & (df.trait==t)
			if s == 'tst':
				# Logical vector for indexation:
				index = df.id_gbs.isin(tst_index[i]) & (df.trait==t)
			# Building the response vector for the subset of data:
			y[t + '_k' + str(i) + '_' + s] = df.y_hat[index]
			# Building feature matrix for the subset of data:
			X[t + '_k' + str(i) + '_' + s] = X_all[index]
			if t == 'drymass':
				X[t + '_k' + str(i) + '_' + s] = X[t + '_k' + str(i) + '_' + s].drop('dap', axis=1)




