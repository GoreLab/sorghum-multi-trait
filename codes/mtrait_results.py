
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
cv2_type = ['cv2-30~45', 'cv2-30~60', 'cv2-30~75', 'cv2-30~90', 'cv2-30~105',
			'cv2-30~only', 'cv2-45~only', 'cv2-60~only', 'cv2-75~only', 'cv2-90~only', 'cv2-105~only']

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

# Reading cv2 files for height:
for t in range(len(cv2_type)):
	for i in range(len(cv_split)):
		for j in range(len(cv1_fold)):
			# Creating the suffix of the file name for the cv2 data case:
			index = cv2_type[t] + '_height_' + cv_split[i]
			# Reading data:
			y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
			X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)

# Reading cv2 file just related to drymass data used in the pleiotropic model:
index = 'cv2_drymass_trn'
y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)


#--------------------------------Computing predictions for the CV1 scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv1 = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Computing predictions for the BN model:
for s in trait_set:
	# Computing predictions for drymass:
	if s=='drymass':
		for j in cv1_fold:
			# Setting the directory:
			os.chdir(prefix_out + 'outputs/cross_validation/BN/cv1/drymass/' + j)
			# Loading stan fit object and model:
			with open("output_bn_fit_0.pkl", "rb") as f:
			    data_dict = pickle.load(f)
			# Indexing the fit object and model
			out = data_dict['fit'].extract()
			# Computing the posterior means:
			mu = out['mu'].mean(axis=0)
			alpha = out['alpha'].mean(axis=0)
			# Index and subsetting the feature matrix:
			index1 = 'cv1_drymass_' + j + '_tst'
			index2 = 'bn_cv1_drymass'
			X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
			# Prediction:
			tmp = mu + X_tmp.dot(alpha)
			# Storing prediction:
			if j=='k0':
				y_pred_cv1[index2] = tmp
			if j!='k0':
				y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=0)
	# Computing predictions for height:
	if s=='height':
		for d in range(len(dap_group)):
			for j in cv1_fold:
				# Setting the directory:
				os.chdir(prefix_out + 'outputs/cross_validation/BN/cv1/height/' + j)
				# Loading stan fit object and model:
				with open("output_bn_fit_" + str(d) + ".pkl", "rb") as f:
				    data_dict = pickle.load(f)
				# Indexing the fit object and model
				out = data_dict['fit'].extract()
				# Computing the posterior means:
				mu = out['mu'].mean(axis=0)
				alpha = out['alpha'].mean(axis=0)
				# Index and subsetting the feature matrix:
				index1 = 'cv1_height_' + j + '_tst'
				index2 = 'bn_cv1_height_' + dap_group[d]
				X_tmp = X[index1][X[index1].iloc[:,0] == int(dap_group[d])]
				X_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
				# Prediction:
				tmp = mu + X_tmp.dot(alpha)
				# Storing prediction:
				if j=='k0':
					y_pred_cv1[index2] = tmp
				if j!='k0':
					y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=0)

# Computing predictions for the PBN model:
for d in range(len(dap_group)):
	for j in cv1_fold:
		# Setting the directory:
		os.chdir(prefix_out + 'outputs/cross_validation/PBN/cv1/drymass-height/' + j)
		# Loading stan fit object and model:
		with open("output_pbn_fit_" + str(d) + ".pkl", "rb") as f:
		    data_dict = pickle.load(f)
		# Indexing the fit object and model
		out = data_dict['fit'].extract()
		# Computing the posterior means:
		mu_0 = out['mu_0'].mean(axis=0)
		mu_1 = out['mu_1'].mean(axis=0)
		alpha_0 = out['alpha_0'].mean(axis=0)
		alpha_1 = out['alpha_1'].mean(axis=0)
		eta_0 = out['eta_0'].mean(axis=0)
		eta_1 = out['eta_1'].mean(axis=0)
		# Index and subsetting the feature matrix:
		index1_0 = 'cv1_drymass_' + j + '_tst'
		index1_1 = 'cv1_height_' + j + '_tst'
		index2_0 = 'pbn_cv1_drymass_' + dap_group[d]
		index2_1 = 'pbn_cv1_height_' + dap_group[d]
		X_tmp_0 = X[index1_0]
		X_tmp_0 = X_tmp_0.drop(X_tmp_0.columns[0], axis=1)
		X_tmp_1 = X[index1_1][X[index1_1].iloc[:,0] == int(dap_group[d])]
		X_tmp_1 = X_tmp_1.drop(X_tmp_1.columns[0], axis=1)
		# Prediction:
		tmp_0 = mu_0 + X_tmp_0.dot(alpha_0 + eta_0)
		tmp_1 = mu_1 + X_tmp_1.dot(alpha_1 + eta_1)
		# Storing prediction:
		if j=='k0':
			y_pred_cv1[index2_0] = tmp_0
			y_pred_cv1[index2_1] = tmp_1
		if j!='k0':
			y_pred_cv1[index2_0] = pd.concat([y_pred_cv1[index2_0], tmp_0], axis=0)
			y_pred_cv1[index2_1] = pd.concat([y_pred_cv1[index2_1], tmp_1], axis=0)

# Ensambling predictions for drymass:
tmp = []
for i in range(len(dap_group)):
	tmp.append(y_pred_cv1['pbn_cv1_drymass_' + dap_group[i]])

# Storing:
y_pred_cv1['pbn_cv1_drymass'] = pd.DataFrame(np.mean(np.vstack(tmp), axis=0), index=y_pred_cv1['pbn_cv1_drymass_30'].index)

# Removing predictions that will not be used anymore:
for d in range(len(dap_group)):
	y_pred_cv1.pop('pbn_cv1_drymass_' + dap_group[d])

# Computing predictions for the DBN model:
for j in cv1_fold:
	# Setting the directory:
	os.chdir(prefix_out + 'outputs/cross_validation/DBN/cv1/height/' + j)
	# Loading stan fit object and model:
	with open("output_dbn-0~6.pkl", "rb") as f:
	    data_dict = pickle.load(f)
	# Indexing the fit object and model
	out = data_dict['fit'].extract()
	for d in range(len(dap_group)):
		# Computing the posterior means:
		beta = out['beta'].mean(axis=0)
		alpha = out['alpha_' + str(d)].mean(axis=0)
		# Index and subsetting the feature matrix:
		index1 = 'cv1_height_' + j + '_tst'
		index2 = 'dbn_cv1_height_' + dap_group[d]
		X_tmp = X[index1][X[index1].iloc[:,0] == int(dap_group[d])]
		Z_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
		X_tmp = X_tmp.iloc[:,0]
		# Prediction:
		tmp = X_tmp * beta  + Z_tmp.dot(alpha)
		# Storing prediction:
		if j=='k0':
			y_pred_cv1[index2] = tmp
		if j!='k0':
			y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=0)


#--------------------------------Computing predictions for the CV2 scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv2 = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105']  

# Computing predictions for the BN model:
for d in range(len(dap_group)):
	# Setting the directory:
	os.chdir(prefix_out + 'outputs/cross_validation/BN/cv2-' + dap_group[d] + '~only/height/')
	# Loading stan fit object and model:
	with open("output_bn_fit_0.pkl", "rb") as f:
		data_dict = pickle.load(f)
	# Indexing the fit object and model
	out = data_dict['fit'].extract()
	# Computing the posterior means:
	mu = out['mu'].mean(axis=0)
	alpha = out['alpha'].mean(axis=0)
	# Index and subsetting the feature matrix:
	index1 = 'cv2-' + dap_group[d] + '~only_height_tst'
	index2 = 'bn_cv2_height_' + dap_group[d]
	X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
	# Prediction:
	y_pred_cv2[index2] = mu + X_tmp.dot(alpha)

# Computing predictions for the PBN model:
for d in range(len(dap_group)):
	# Setting the directory:
	os.chdir(prefix_out + 'outputs/cross_validation/PBN/cv2-' + dap_group[d] + '~only/drymass-height/')
	# Loading stan fit object and model:
	with open("output_pbn_fit_0.pkl", "rb") as f:
		data_dict = pickle.load(f)
	# Indexing the fit object and model
	out = data_dict['fit'].extract()
	# Computing the posterior means:
	mu = out['mu_1'].mean(axis=0)
	alpha = out['alpha_1'].mean(axis=0)
	eta = out['eta_1'].mean(axis=0)
	# Index and subsetting the feature matrix:
	index1 = 'cv2-' + dap_group[d] + '~only_height_tst'
	index2 = 'pbn_cv2_height_' + dap_group[d]
	X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
	# Prediction:
	y_pred_cv2[index2] = mu + X_tmp.dot(alpha + eta)

# Different DAP measures:
cv2_type = ['cv2-30~45', 'cv2-30~60', 'cv2-30~75', 'cv2-30~90', 'cv2-30~105']
dap_index = ['0~1', '0~2', '0~3', '0~4', '0~5']

# Computing predictions for the DBN model:
for c in range(len(cv2_type)):
	# Setting the directory:
	os.chdir(prefix_out + 'outputs/cross_validation/DBN/' + cv2_type[c] + '/height/')
	# Loading stan fit object and model:
	with open("output_dbn-" + dap_index[c] + ".pkl", "rb") as f:
		data_dict = pickle.load(f)
	# Indexing the fit object and model
	out = data_dict['fit'].extract()
	# Getting the last time point used for training:
	upper = dap_index[c].split('~')[1]
	# Computing the posterior means:
	beta = out['beta'].mean(axis=0)
	alpha = out['alpha_' + upper].mean(axis=0)
	# Index and subsetting the feature matrix:
	index1 = cv2_type[c] +'_height_tst'
	index2 = 'dbn_cv2_height_' + cv2_type[c].split('-')[1]
	Z_tmp = X[index1].drop(X[index1].columns[0], axis=1)
	X_tmp = X[index1].iloc[:,0]
	# Prediction:
	y_pred_cv2[index2] = X_tmp * beta + Z_tmp.dot(alpha)



