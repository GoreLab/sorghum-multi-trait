
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
for t in df.trait.unique():
	for i in range(n_fold):
		y[t + '_k' + i] = df.y_hat[df.id_gbs==trn_index[i]]


####3


tmp = pd.get_dummies(df.id_gbs[df.trait=='height'])

X['height'] = pd.concat([X['height'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)
