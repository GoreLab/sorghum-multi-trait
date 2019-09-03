#------------------------------------------------Modules-----------------------------------------------------#

# Load libraries:
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
import argparse
from itertools import chain
parser = argparse.ArgumentParser()


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Get flags:
parser.add_argument("-dpath", "--dpath", dest = "dpath", help="The path of the folder with the raw data")
parser.add_argument("-opath", "--opath", dest = "opath", help="The path of the folder to receive outputs")

# Parse the paths:
args = parser.parse_args()

# Subset arguments:
DATA_PATH = args.dpath
OUT_PATH = args.opath
# DATA_PATH = '/workdir/jp2476/raw_data_sorghum-multi-trait'
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'


#------------------------------------Load Adjusted Means and plotting----------------------------------------#

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Change class of the dap:
df.dap = df.dap.fillna(0).astype(int)

# Read marker binned matrix:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

# Set the directory to store processed data:
os.chdir(DATA_PATH + "/raw_data")

# Read the file with the annotation of the population structure of the lines:
pop_struc = pd.read_csv("population_structure.txt", header = 0, sep='\t')

# Read the file with the annotation of the population structure of the lines:
pop_struc = pd.read_csv("population_structure.txt", header = 0, sep='\t')
pop_struc = pop_struc.rename(columns={'ID':'id_gbs', 'CATEGORY':'id_pop'})

#----------------------------Split data into groups for 5th-fold cross-validation----------------------------#

# Initialize dictionary to receive index stratified by population type:
pop_folds = dict()

# Number of folds:
n_fold = 5

# Initialize lists:
for counter in range(n_fold):
	pop_folds['k' + str(counter)] = []

# Split the data into subsets stratified by population:
for pop in pop_struc.id_pop.unique():
	counter=0
	# Create five folds:
	kf = KFold(n_splits=n_fold, shuffle=True, random_state=(1234+counter))
	# Get the splits:
	index = kf.split(pop_struc.id_gbs[pop_struc.id_pop == pop])
	# Get the indexes:
	for _, fold in index:
		pop_folds['k' + str(counter)].append(pop_struc.id_gbs[pop_struc.id_pop == pop].iloc[fold].tolist())
		counter+=1

# Unnest the lists:
for counter in range(n_fold):
	pop_folds['k' + str(counter)] = list(chain.from_iterable(pop_folds['k' + str(counter)]))

# Initialize lists to receive the random indexes:
trn_index = []
tst_index = []
 
# Build sets:
trn_index.append(pop_folds['k0'] + pop_folds['k1'] + pop_folds['k2'] + pop_folds['k3']) 
tst_index.append(pop_folds['k4'])

trn_index.append(pop_folds['k0'] + pop_folds['k1'] + pop_folds['k2'] + pop_folds['k4']) 
tst_index.append(pop_folds['k3'])

trn_index.append(pop_folds['k0'] + pop_folds['k1'] + pop_folds['k3'] + pop_folds['k4']) 
tst_index.append(pop_folds['k2'])

trn_index.append(pop_folds['k0'] + pop_folds['k2'] + pop_folds['k3'] + pop_folds['k4']) 
tst_index.append(pop_folds['k1'])

trn_index.append(pop_folds['k1'] + pop_folds['k2'] + pop_folds['k3'] + pop_folds['k4']) 
tst_index.append(pop_folds['k0'])

# Create dictionary with the data from the first cross-validation scheme:
y = dict()
X = dict()

# Build the response vector with all entries:
y_all = df.y_hat

# Build the feature matrix using all data set entries:
X_all = pd.get_dummies(df['id_gbs']) 
X_all = X_all.dot(W_bin.loc[X_all.columns])
X_all = pd.concat([df.dap, X_all], axis=1)

# Create different sets:
sets = ['trn', 'tst']

# Build the sets of the data for the 5-fold cross-validation scheme:
for s in sets:
	for t in df.trait.unique():
		for i in range(n_fold):
			# Key index for building the dictionary:
			key_index = 'cv5f_' + t + '_k' + str(i) + '_' + s
			if s == 'trn':
				# Logical vector for indexation:
				index = df.id_gbs.isin(trn_index[i]) & (df.trait==t)
			if s == 'tst':
				# Logical vector for indexation:
				index = df.id_gbs.isin(tst_index[i]) & (df.trait==t)
			# Build the response vector for the subset of data:
			y[key_index] = df.y_hat[index]
			# Build feature matrix for the subset of data:
			X[key_index] = X_all[index]
			if t == 'drymass':
				X[key_index] = X[key_index].replace(to_replace=0, value=1) \
										   .rename(columns={'dap': 'mu_index'})

# Name of the forward-chaining cross-validation schemes:
fcv_types = ['fcv-30~45', 'fcv-30~60', 'fcv-30~75', 'fcv-30~90', 'fcv-30~105']

# Build the sets of the data for the forward-chaining cross-validation schemes for the DBN model:
for c in fcv_types:
	for s in sets:
		# Key index for building the dictionary:
		key_index = c + '_height_' + s
		# Get the upper index of the dap:
		upper_index = int(c.split('~')[1])
		if s == 'trn':
			# Logical vector for indexation:
			index = (df.trait=='height') & (df.dap<=upper_index)
		if s == 'tst':
			# Logical vector for indexation:
			index = (df.trait=='height') & (df.dap>upper_index)
		# Build the response vector for the subset of data:
		y[key_index] = df.y_hat[index]
		# Build feature matrix for the subset of data:
		X[key_index] = X_all[index]

# Get DAP groups:
dap_group = df.dap.unique()[1:7]

# Build the sets of the data for the forward-chaining cross-validation schemes for the PBN and BN models using all data:
for d in dap_group:
	for s in sets:
		# Key index for mapping data into dictionary:
		key_index = 'fcv-' + str(d) + '~only' + '_height_' + s
		if s == 'trn':
			# Logical vector for indexation:
			index = (df.trait=='height') & (df.dap==d)
		if s == 'tst':
			# Logical vector for indexation:
			index = (df.trait=='height') & (df.dap!=d)
		# Build the response vector for the subset of data:
		y[key_index] = df.y_hat[index]
		# Build feature matrix for the subset of data:
		X[key_index] = X_all[index]

# Create the sets only for the PBN model with drymass data together with height data:
key_index = 'fcv_drymass_trn'
y[key_index] = df.y_hat[df.trait=='drymass']
X[key_index] = X_all[df.trait=='drymass']


#------------------------------------Save different subsets of the data--------------------------------------#

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Save cross-validation data:
for i in y.keys():
	y[i].to_csv('y_' + i + '.csv', header='y_hat')
	X[i].to_csv('x_' + i + '.csv')
