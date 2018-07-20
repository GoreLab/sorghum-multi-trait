
#------------------------------------------------Modules-----------------------------------------------------#

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load libraries:
import pandas as pd
import numpy as np
import os
import pickle
import re
import pystan as ps
from scipy.stats.stats import pearsonr
from pprint import pprint as pprint
from pymc3.stats import hpd 
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

# Create a list with the cv1 folds:
cv1_fold = ['k0', 'k1', 'k2', 'k3', 'k4']

# Create a list with the traits set:
trait_set = ['drymass', 'height']

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Set the directory:
os.chdir(prefix_out + 'data/cross_validation/')

# Read cv1 files:
for s in range(len(trait_set)):
  for j in range(len(cv1_fold)):
    # Create the suffix of the file name for the cv1 data case:
    index = 'cv1_' + trait_set[s] + '_' + cv1_fold[j] + '_tst'
    # Read data:
    y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
    X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)

# Read cv2 files for height:
for t in range(len(cv2_type)):
  for j in range(len(cv1_fold)):
    # Create the suffix of the file name for the cv2 data case:
    index = cv2_type[t] + '_height_tst'
    # Read data:
    y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
    X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)


#----------------------------Organizing data into a new format for easy subsetting---------------------------#

# Set directory:
os.chdir(prefix_out + "outputs/first_step_analysis")

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)
df.dap = df.dap.fillna(0).astype(int)

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Initialize dictionary to receive observations for cv1 analysis:
y_obs_cv1 = dict()

# Store observations for height stratified by modelling scenario for cv1 analysis:
for d in dap_group:
  y_obs_cv1['cv1_height_dap:' + d] = df.y_hat[(df.trait=='height') & (df.dap==int(d))] 

# Store observations for drymass stratified by modelling scenario for cv1 analysis:
y_obs_cv1['cv1_drymass'] = df.y_hat[df.trait=='drymass'] 

# Initialize dictionary to receive observations for cv2 analysis:
y_obs_cv2 = dict()

# Store observations for height stratified by modelling scenario for cv2 analysis:
dap_group = ['30', '45', '60', '75', '90', '105']  
for d in dap_group:
  y_obs_cv2['cv2_height_for!trained!on:' + d] = y['cv2-' + d + '~only_height_tst']

# Store observations for height stratified by modelling scenario for cv2 analysis:
dap_group = ['30~45', '30~60', '30~75', '30~90', '30~105']
for d in dap_group:
  y_obs_cv2['cv2_height_for!trained!on:' + d] = y['cv2-' + d + '_height_tst']


#--------------------------------Computing predictions for the CV1 scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv1 = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Compute predictions for the BN model:
for s in trait_set:
  # Compute predictions for drymass:
  if s=='drymass':
    for j in cv1_fold:
      # Set the directory:
      os.chdir(prefix_out + 'outputs/cross_validation/BN/cv1/drymass/' + j)
      # Load stan fit object and model:
      with open("output_bn_fit_0.pkl", "rb") as f:
          data_dict = pickle.load(f)
      # Index the fit object and model
      out = data_dict['fit'].extract()
      # Index and subsetting the feature matrix:
      index1 = 'cv1_drymass_' + j + '_tst'
      index2 = 'bn_cv1_drymass'
      X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
      # Create the indexes for creating a data frame to receive predictions:
      index = ["s_" + str(i) for i in range(out['mu'].size)]
      col_name = X_tmp.index
      # Initialize a matrix to receive the posterior predictions:
      tmp = pd.DataFrame(index=index, columns=col_name)
      # Compute predictions:
      for sim in range(out['mu'].size):
        # Subset parameters:
        mu = out['mu'][sim]
        alpha = out['alpha'][sim,:]
        # Prediction:
        tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values
      # Store prediction:
      if j=='k0':
        y_pred_cv1[index2] = tmp
      if j!='k0':
        y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=1)
  # Compute predictions for height:
  if s=='height':
    for d in range(len(dap_group)):
      for j in cv1_fold:
        # Set the directory:
        os.chdir(prefix_out + 'outputs/cross_validation/BN/cv1/height/' + j)
        # Load stan fit object and model:
        with open("output_bn_fit_" + str(d) + ".pkl", "rb") as f:
            data_dict = pickle.load(f)
        # Index the fit object and model
        out = data_dict['fit'].extract()
        # Index and subsetting the feature matrix:
        index1 = 'cv1_height_' + j + '_tst'
        index2 = 'bn_cv1_height_trained!on!dap:' + dap_group[d]
        X_tmp = X[index1][X[index1].iloc[:,0] == int(dap_group[d])]
        X_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
        # Create the indexes for creating a data frame to receive predictions:
        index = ["s_" + str(i) for i in range(out['mu'].size)]
        col_name = X_tmp.index
        # Initialize a matrix to receive the posterior predictions:
        tmp = pd.DataFrame(index=index, columns=col_name)
        # Compute predictions:
        for sim in range(out['mu'].size):
          # Subset parameters:
          mu = out['mu'][sim]
          alpha = out['alpha'][sim,:]
          # Prediction:
          tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values
        # Store prediction:
        if j=='k0':
          y_pred_cv1[index2] = tmp
        if j!='k0':
          y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=1)

# Compute predictions for the PBN model:
for d in range(len(dap_group)):
  for j in cv1_fold:
    # Set the directory:
    os.chdir(prefix_out + 'outputs/cross_validation/PBN/cv1/drymass-height/' + j)
    # Load stan fit object and model:
    with open("output_pbn_fit_" + str(d) + ".pkl", "rb") as f:
        data_dict = pickle.load(f)
    # Index the fit object and model
    out = data_dict['fit'].extract()
    # Index and subset the feature matrix:
    index1_0 = 'cv1_drymass_' + j + '_tst'
    index1_1 = 'cv1_height_' + j + '_tst'
    index2_0 = 'pbn_cv1_drymass_trained!on!dap:' + dap_group[d]
    index2_1 = 'pbn_cv1_height_trained!on!dap:' + dap_group[d]
    X_tmp_0 = X[index1_0]
    X_tmp_0 = X_tmp_0.drop(X_tmp_0.columns[0], axis=1)
    X_tmp_1 = X[index1_1][X[index1_1].iloc[:,0] == int(dap_group[d])]
    X_tmp_1 = X_tmp_1.drop(X_tmp_1.columns[0], axis=1)
    # Create the indexes for creating a data frame to receive predictions:
    index = ["s_" + str(i) for i in range(out['mu_0'].size)]
    col_name = X_tmp_0.index
    # Initialize a matrix to receive the posterior predictions:
    tmp_0 = pd.DataFrame(index=index, columns=col_name)
    # Create the indexes for creating a data frame to receive predictions:
    index = ["s_" + str(i) for i in range(out['mu_0'].size)]
    col_name = X_tmp_1.index
    # Initialize a matrix to receive the posterior predictions:
    tmp_1 = pd.DataFrame(index=index, columns=col_name)
    # Compute predictions:
    for sim in range(out['mu_0'].size):
      # Subset parameters:
      mu_0 = out['mu_0'][sim]
      mu_1 = out['mu_1'][sim]
      alpha_0 = out['alpha_0'][sim,:]
      alpha_1 = out['alpha_1'][sim,:]
      eta_0 = out['eta_0'][sim,:]
      eta_1 = out['eta_1'][sim,:]
      # Prediction:
      tmp_0.iloc[sim] = (mu_0 + X_tmp_0.dot(alpha_0 + eta_0)).values
      tmp_1.iloc[sim] = (mu_1 + X_tmp_1.dot(alpha_1 + eta_1)).values
    # Store prediction:
    if j=='k0':
      y_pred_cv1[index2_0] = tmp_0
      y_pred_cv1[index2_1] = tmp_1
    if j!='k0':
      y_pred_cv1[index2_0] = pd.concat([y_pred_cv1[index2_0], tmp_0], axis=1)
      y_pred_cv1[index2_1] = pd.concat([y_pred_cv1[index2_1], tmp_1], axis=1)

# List drymass predictions for ensambling:
tmp = []
for i in range(len(dap_group)):
  tmp.append(y_pred_cv1['pbn_cv1_drymass_trained!on!dap:' + dap_group[i]])

# Ensamble predictions for drymass:
y_pred_cv1['pbn_cv1_drymass_ensambled'] = (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]) / 7

# Remove predictions that will not be used anymore:
for d in range(len(dap_group)):
  y_pred_cv1.pop('pbn_cv1_drymass_trained!on!dap:' + dap_group[d])

# Compute predictions for the DBN model:
for j in cv1_fold:
  # Set the directory:
  os.chdir(prefix_out + 'outputs/cross_validation/DBN/cv1/height/' + j)
  # Load stan fit object and model:
  with open("output_dbn-0~6.pkl", "rb") as f:
      data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  for d in range(len(dap_group)):
    # Index and subset the feature matrix:
    index1 = 'cv1_height_' + j + '_tst'
    index2 = 'dbn_cv1_height_trained!on!dap:' + dap_group[d]
    X_tmp = X[index1][X[index1].iloc[:,0] == int(dap_group[d])]
    Z_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
    # Create the indexes for creating a data frame to receive predictions:
    index = ["s_" + str(i) for i in range(out['mu_' + str(d)].size)]
    col_name = X_tmp.index
    # Initialize a matrix to receive the posterior predictions:
    tmp = pd.DataFrame(index=index, columns=col_name)
    # Compute predictions:
    for sim in range(out['mu_0'].size):
      # Subset parameters:
      mu = out['mu_' + str(d)][sim]
      alpha = out['alpha_' + str(d)][sim,:]
      # Prediction:
      tmp.iloc[sim] = (mu + Z_tmp.dot(alpha)).values
    # Store prediction:
    if j=='k0':
      y_pred_cv1[index2] = tmp
    if j!='k0':
      y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=1)


#--------------------------------Computing predictions for the CV2 scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv2 = dict()

# Initialize list to receive the expectaions:
expect_cv2 = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105']  

# Compute predictions for the BN model:
for d in range(len(dap_group)):
  # Set the directory:
  os.chdir(prefix_out + 'outputs/cross_validation/BN/cv2-' + dap_group[d] + '~only/height/')
  # Load stan fit object and model:
  with open("output_bn_fit_0.pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  # Index and subset the feature matrix:
  index1 = 'cv2-' + dap_group[d] + '~only_height_tst'
  index2 = 'bn_cv2_height_trained!on!dap:' + dap_group[d]
  X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
  # Create the indexes for creating a data frame to receive predictions:
  index = ["s_" + str(i) for i in range(out['mu'].size)]
  col_name = X_tmp.index
  # Initialize a matrix to receive the posterior predictions:
  tmp = pd.DataFrame(index=index, columns=col_name)
  # Compute predictions:
  for sim in range(out['mu'].size):
    # Subset parameters:
    mu = out['mu'][sim]
    alpha = out['alpha'][sim,:]
    # Prediction:
    tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values
  # Store predictions:
  y_pred_cv2[index2] = tmp
  # Store expectations:
  expect_cv2[index2] = pd.DataFrame(out['expectation'], index=index, columns=df.index[df.dap==int(dap_group[d])])

# Compute predictions for the PBN model:
for d in range(len(dap_group)):
  # Set the directory:
  os.chdir(prefix_out + 'outputs/cross_validation/PBN/cv2-' + dap_group[d] + '~only/drymass-height/')
  # Load stan fit object and model:
  with open("output_pbn_fit_0.pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fitted object and model
  out = data_dict['fit'].extract()
  # Index and subset the feature matrix:
  index1 = 'cv2-' + dap_group[d] + '~only_height_tst'
  index2 = 'pbn_cv2_height_trained!on!dap:' + dap_group[d]
  X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
  # Create the indexes for creating a data frame to receive predictions:
  index = ["s_" + str(i) for i in range(out['mu_1'].size)]
  col_name = X_tmp.index
  # Initialize a matrix to receive the posterior predictions:
  tmp = pd.DataFrame(index=index, columns=col_name)
  # Compute predictions:
  for sim in range(out['mu_1'].size):
    # Subset parameters:
    mu = out['mu_1'][sim]
    alpha = out['alpha_1'][sim,:]
    eta = out['eta_1'][sim,:]
    # Prediction:
    tmp.iloc[sim] = (mu + X_tmp.dot(alpha + eta)).values
  # Store predictions:
  y_pred_cv2[index2] = tmp
  # Store expectations:
  expect_cv2[index2] = pd.DataFrame(out['expectation_1'], index=index, columns=df.index[df.dap==int(dap_group[d])])

# Different DAP measures:
cv2_type = ['cv2-30~45', 'cv2-30~60', 'cv2-30~75', 'cv2-30~90', 'cv2-30~105']
dap_index = ['0~1', '0~2', '0~3', '0~4', '0~5']

# Compute predictions for the DBN model:
for c in range(len(cv2_type)):
  # Set the directory:
  os.chdir(prefix_out + 'outputs/cross_validation/DBN/' + cv2_type[c] + '/height/')
  # Load stan fit object and model:
  with open("output_dbn-" + dap_index[c] + ".pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  # Get the last time point used for training:
  upper = dap_index[c].split('~')[1]
  # Index and subset the feature matrix:
  index1 = cv2_type[c] +'_height_tst'
  index2 = 'dbn_cv2_height_trained!on!dap:' + cv2_type[c].split('-')[1]
  Z_tmp = X[index1].drop(X[index1].columns[0], axis=1)
  # Create the indexes for creating a data frame to receive predictions:
  index = ["s_" + str(i) for i in range(out['mu_0'].size)]
  col_name = Z_tmp.index
  # Initialize a matrix to receive the posterior predictions:
  tmp = pd.DataFrame(index=index, columns=col_name)
  # Compute predictions:
  for sim in range(out['mu_0'].size):
    # Subset parameters:
    mu = out['mu_' + upper][sim]
    alpha = out['alpha_' + upper][sim,:]
    # Prediction:
    tmp.iloc[sim] = (mu + Z_tmp.dot(alpha)).values
  # Prediction:
  y_pred_cv2[index2] = tmp
  # Store expectations:
  expect_cv2[index2] = pd.DataFrame(out['expectation_'  + upper], index=index, columns=df.index[df.dap==int(dap_group[int(upper)])])


#--------------------------------Computing predictions for the CV3 scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv3 = dict()

# Create a list with the cv1 folds:
cv3_fold = ['k0', 'k1', 'k2', 'k3', 'k4']

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105']  

# Predict the drymass values under the PBN model:
for d in range(len(dap_group)):
  for j in cv3_fold:
    # Set the directory:
    os.chdir(prefix_out + 'outputs/cross_validation/PBN/cv3-' + dap_group[d] + '~only/drymass-height/' + j)
    # Load stan fit object and model:
    with open("output_pbn_fit_0.pkl", "rb") as f:
        data_dict = pickle.load(f)
    # Index the fit object and model
    out = data_dict['fit'].extract()
    # Index and subsetting the feature matrix:
    index1 = 'cv1_drymass_' + j + '_tst'
    index2 = 'pbn_cv3_drymass-height_trained!on!dap:' + dap_group[d]
    X_tmp = X[index1].drop(X[index1].columns[0], axis=1)
    # Create the indexes for creating a data frame to receive predictions:
    index = ["s_" + str(i) for i in range(out['mu_0'].size)]
    col_name = X_tmp.index
    # Initialize a matrix to receive the posterior predictions:
    tmp = pd.DataFrame(index=index, columns=col_name)
    # Compute predictions:
    for sim in range(out['mu_0'].size):
      # Subset parameters:
      mu = out['mu_0'][sim]
      alpha = out['alpha_0'][sim,:]
      # Prediction:
      tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values
    # Store prediction:
    if j=='k0':
      y_pred_cv3[index2] = tmp
    if j!='k0':
      y_pred_cv3[index2] = pd.concat([y_pred_cv3[index2], tmp], axis=1)
    print('DAP: {}, fold: {}'.format(dap_group[d], j))


#-----------------------------Compute prediction accuracies for the CV1 scheme-------------------------------#

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105']  

# Accuracies for cv1 from the height prediction:
for d in dap_group:
  index = y_pred_cv1['bn_cv1_height_trained!on!dap:' + d].columns
  np.round(pearsonr(y_pred_cv1['bn_cv1_height_trained!on!dap:' + d].mean(axis=0), y_obs_cv1['cv1_height_dap:' + d][index])[0],2)

for d in dap_group:
  index = y_pred_cv1['pbn_cv1_height_trained!on!dap:' + d].columns
  np.round(pearsonr(y_pred_cv1['pbn_cv1_height_trained!on!dap:' + d].mean(axis=0), y_obs_cv1['cv1_height_dap:' + d][index])[0],2)

for d in dap_group:
  index = y_pred_cv1['dbn_cv1_height_trained!on!dap:' + d].columns
  np.round(pearsonr(y_pred_cv1['dbn_cv1_height_trained!on!dap:' + d].mean(axis=0), y_obs_cv1['cv1_height_dap:' + d][index])[0],2)

# Accuracies for cv1 from the drymass prediction:
index = y_pred_cv1['bn_cv1_drymass'].columns
np.round(pearsonr(y_pred_cv1['bn_cv1_drymass'].mean(axis=0), y_obs_cv1['cv1_drymass'][index])[0],2)

index = y_pred_cv1['pbn_cv1_drymass_ensambled'].columns
np.round(pearsonr(y_pred_cv1['pbn_cv1_drymass_ensambled'].mean(axis=0), y_obs_cv1['cv1_drymass'][index])[0],2)


# Accuracies for cv3:
for k in list(y_pred_cv3.keys()):
  index = y_obs_cv1['cv1_drymass'].index
  np.round(pearsonr(y_pred_cv3[k].mean(axis=0)[index], y_obs_cv1['cv1_drymass'])[0],2)


#-----------------------------Compute prediction accuracies for the CV2 scheme-------------------------------#


# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive the accuracy matrices:
cor_dict = dict()

# Compute correlation for the Baysian Network and Pleiotropic Bayesian Network model under CV2 scheme:
for k in model_set:
  # Create an empty correlation matrix:
  cor_tmp = np.empty([len(dap_group)]*2)
  cor_tmp[:] = np.nan
  for i in range(len(dap_group[:-1])):
    # Subsetting predictions for correlation computation:
    y_pred_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]].mean(axis=0)
    y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group[i]].y_hat
    for j in range(len(dap_group)):
      # Conditional to compute correlation just forward in time:
      if (j>i):
        # Subset indexes for subsetting the data:
        subset = df[df.dap==int(dap_group[j])].index
        # Build correlation matrix for the Bayesian Network model:
        cor_tmp[i, j] = np.round(pearsonr(y_pred_tmp[subset], y_obs_tmp[subset])[0],4)
  # Store the computed correlation matrix for the Bayesian network model under CV2 scheme:
  cor_dict['cv2_' + k] = cor_tmp

# Print predictive accuracies
print(cor_dict['cv2_bn'])
print(cor_dict['cv2_pbn'])

# Store into a list different DAP values intervals:
dap_group1 = ['30~45', '30~60', '30~75', '30~90', '30~105']
dap_group2 = ['30', '45', '60', '75', '90', '105', '120']

# Create an empty correlation matrix:
cor_tmp = np.empty([len(dap_group)]*2)
cor_tmp[:] = np.nan

# Compute correlation for the Dynamic Bayesian Network model under CV2 scheme:
for i in range(len(dap_group1)):
  # Subset predictions for correlation computation:
  y_pred_tmp = y_pred_cv2['dbn_cv2_height_trained!on!dap:' + dap_group1[i]].mean(axis=0)
  y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group1[i]].y_hat
  for j in range(len(dap_group2)):    
    # Get the upper bound of the interval:
    upper = int(dap_group1[i].split('~')[1])
    # Conditional to compute correlation just forward in time:
    if (int(dap_group2[j])>upper):
      # Subset indexes for subsetting the data:
      subset = df[df.dap==int(dap_group2[j])].index
      # Build correlation matrix for the Bayesian Network model:
      cor_tmp[i, j] = np.round(pearsonr(y_pred_tmp[subset], y_obs_tmp[subset])[0],4)

# Store the computed correlation matrix for the Bayesian network model under CV2 scheme:
cor_dict['cv2_dbn'] = cor_tmp

# Builda mask just to filter accuracies common across models:
mask = np.isnan(cor_dict['cv2_dbn'])

# Eliminate values not shared across all models:
cor_dict['cv2_bn'][mask] = np.nan
cor_dict['cv2_pbn'][mask] = np.nan

# Eliminate rows and columns without any correlation value:
cor_dict['cv2_bn'] = cor_dict['cv2_bn'][0:5,2:7]
cor_dict['cv2_pbn'] = cor_dict['cv2_pbn'][0:5,2:7]
cor_dict['cv2_dbn'] = cor_dict['cv2_dbn'][0:5,2:7]

# Print predictive accuracies
print(cor_dict['cv2_bn'])
print(cor_dict['cv2_pbn'])
print(cor_dict['cv2_dbn'])

# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive probabilities:
prob_dict = dict()

for k in model_set:
  for i in range(len(dap_group[:-1])):
    # Subset predictions and observations for probability computation:
    y_pred_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]
    y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group[i]].y_hat
    # Compute probability across DAP measures:
    for j in range(len(dap_group)):
      # Conditional to compute probability just forward in time:
      if (j>i):
        # Subset indexes for subsetting the data:
        subset = df[df.dap==int(dap_group[j])].index
        # Get the number of selected individuals for 20% selection intensity:
        n_selected = int(y_obs_tmp[subset].size * 0.2)
        # Build the indexes for computing the coeficient of coincidence:
        rank_obs = np.argsort(y_obs_tmp[subset])[::-1].index[0:n_selected]
        # Selected lines:
        top_lines_obs = df.id_gbs[rank_obs]
        # Vector for storing the indicators:
        ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
        # Build probability matrix for the Bayesian Network model:
        for sim in range(y_pred_tmp.shape[0]):
          # Top predicted lines:
          rank_pred = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1][0:n_selected]
          top_lines_pred = df.id_gbs[y_pred_tmp[subset].iloc[sim].iloc[rank_pred].index]
          # Compute the indicator of coincidence in the top 20%
          ind_tmp = top_lines_pred.isin(top_lines_obs)
          index_tmp = ind_tmp.iloc[np.where(ind_tmp)].index
          ind_vec.iloc[sim] = y_pred_tmp[subset].iloc[sim].index.isin(index_tmp)
        # Index to store probabilties into dictionary:
        index = k + '_' + dap_group[i] + '_' + dap_group[j]
        # Compute probability:
        prob_dict[index]=ind_vec.mean(axis=0)
        print('Model: {}, DAP_i: {}, DAP_j: {}'.format(k, dap_group[i], dap_group[j]))

# Store into a list different DAP values intervals:
dap_group1 = ['30~45', '30~60', '30~75', '30~90', '30~105']
dap_group2 = ['30', '45', '60', '75', '90', '105', '120']

# Compute probabilities for the Dynamic Bayesian network model under CV2 scheme:
for i in range(len(dap_group1)):
  # Subset predictions for correlation computation:
  y_pred_tmp = y_pred_cv2['dbn_cv2_height_trained!on!dap:' + dap_group1[i]]
  y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group1[i]].y_hat
  for j in range(len(dap_group2)):    
    # Get the upper bound of the interval:
    upper = int(dap_group1[i].split('~')[1])
    # Conditional to compute correlation just forward in time:
    if (int(dap_group2[j])>upper):
      # Subset indexes for subsetting the data:
      subset = df[df.dap==int(dap_group2[j])].index
      # Get the number of selected individuals for 20% selection intensity:
      n_selected = int(y_obs_tmp[subset].size * 0.2)
      # Build the indexes for computing the coeficient of coincidence:
      rank_obs = np.argsort(y_obs_tmp[subset])[::-1].index[0:n_selected]
      # Selected lines:
      top_lines_obs = df.id_gbs[rank_obs]
      # Vector for storing the indicators:
      ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
      # Build probability matrix for the Bayesian Network model:
      for sim in range(y_pred_tmp.shape[0]):
        # Top predicted lines:
        rank_pred = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1][0:n_selected]
        top_lines_pred = df.id_gbs[y_pred_tmp[subset].iloc[sim].iloc[rank_pred].index]
        # Compute the indicator of coincidence in the top 20%
        ind_tmp = top_lines_pred.isin(top_lines_obs)
        index_tmp = ind_tmp.iloc[np.where(ind_tmp)].index
        ind_vec.iloc[sim] = y_pred_tmp[subset].iloc[sim].index.isin(index_tmp)
      # Index to store probabilties into dictionary:
      index = 'dbn_' + dap_group1[i] + '_' + dap_group2[j]
      # Compute probability:
      prob_dict[index]=ind_vec.mean(axis=0)
      print('Model: dbn, DAP_i: {}, DAP_j: {}'.format(dap_group1[i], dap_group2[j]))

# Set directory:
os.chdir(prefix_proj + 'plots/cv/heatplot')

# List of models to use:
model_set = ['bn', 'pbn', 'dbn']

# Generate accuracy heatmaps:
for i in model_set:
  # Labels for plotting the heatmap for the Pleiotropic Bayesian Network or Bayesian Network:
  if (i=='bn') | (i=='pbn'):
    labels_axis0 = ['DAP 45*', 'DAP 60*', 'DAP 75*', 'DAP 90*', 'DAP 105*']
  # Labels for plotting the heatmap for the Dynamic Bayesian model:
  if i=='dbn':
    labels_axis0 = ['DAP 30:45*', 'DAP 30:60*', 'DAP 30:75*', 'DAP 30:90*', 'DAP 30:105*']
  # Labels for plotting the heatmap:
  labels_axis1 = ['DAP 60', 'DAP 75', 'DAP 90', 'DAP 105', 'DAP 120']
  # Heat map of the adjusted means across traits:
  heat = sns.heatmap(np.flip(np.flip(cor_dict['cv2_' + i],axis=1), axis=0),
             linewidths=0.25,
             vmin=0.3,
             vmax=1,
             annot=True,
             annot_kws={"size": 18},
             xticklabels=labels_axis0 ,
             yticklabels=labels_axis1)
  heat.set_ylabel('')    
  heat.set_xlabel('')
  plt.xticks(rotation=25)
  plt.yticks(rotation=45)
  plt.savefig("heatplot_cv2_" + i + "_accuracy.pdf", dpi=150)
  plt.savefig("heatplot_cv2_" + i + "_accuracy.png", dpi=150)
  plt.clf()

# Set directory:
os.chdir(prefix_proj + 'plots/cv/probplot')

# DAP groups used for plotting
dap_group = ['45', '60', '75', '90', '105']

# Generating panel plot:
for i in range(len(dap_group)):
  # Subset probability for plotting:
  prob=prob_dict['dbn_30~' + dap_group[i] + '_120']
  if i==0:
    # Get the order of the probabilities:
    order_index = np.argsort(prob)[::-1]
    # Individuais displaying probability higher then 80%:
    mask = prob.iloc[order_index] > 0.8
  # Subset probability for plotting:
  p1 = prob.iloc[order_index][mask].plot.barh(color='red', figsize=(10,12))
  p1.set(yticklabels=df.id_gbs[mask.index])
  p1.tick_params(axis='y', labelsize=7)
  p1.tick_params(axis='x', labelsize=12)
  plt.xlabel('Top 20% rank joint probabilities', fontsize=20)
  plt.ylabel('Sorghum inbred lines', fontsize=20)
  plt.xlim(0.5, 1)
  plt.savefig('probplot_cv2_' + 'dbn_30~' + dap_group[i] + '_120.pdf', dpi=150)
  plt.savefig('probplot_cv2_' + 'dbn_30~' + dap_group[i] + '_120.png', dpi=150)
  plt.clf()

##########


#-----------------Compute coincidence index for dry biomass selection using height adjusted means------------#

# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Compute coincidence index for the Bayesian network and Pleiotropic Bayesian Network model:
for k in model_set:
  for i in range(len(dap_group[:-1])):
    # Subset expectations and observations for coincidence index computation:
    expect_tmp = expect_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]
    y_obs_tmp = df[df.trait=='drymass'].y_hat
    # Get the number of selected individuals for 20% selection intensity:
    n_selected = int(y_obs_tmp.size * 0.2)
    # Build the indexes for computing the coincidence index:
    rank_obs = np.argsort(y_obs_tmp)[::-1][0:n_selected].index
    # Selected lines:
    top_lines_obs = df[df.trait=='drymass'].id_gbs[rank_obs]
    # Vector for coincidence indexes:
    ci_post_tmp = pd.DataFrame(index=expect_tmp.index, columns=['ci'])
    # Compute the coincidence index:
    for sim in range(ci_post_tmp.shape[0]):
      # Build the indexes for computing the coincidence index:
      rank_pred = np.argsort(expect_tmp.iloc[sim])[::-1][0:n_selected]
      # Selected lines:
      top_lines_pred = df.id_gbs[expect_tmp.iloc[sim].iloc[rank_pred].index]
      # Compute coincidence index in the top 20% or not: 
      ci_post_tmp.iloc[sim] = top_lines_pred.isin(top_lines_obs).mean(axis=0)
    if (k==model_set[0]) & (i==0):
      # Store the coincidence index:
      ci_post = pd.DataFrame(columns=['post', 'model', 'dap'])
      ci_post['model'] = np.repeat(k.upper(), ci_post_tmp.shape[0])
      ci_post['dap'] = np.repeat(('DAP ' + dap_group[i] + '*'), ci_post_tmp.shape[0])
      ci_post['post'] = ci_post_tmp.ci.values
    else:
      # Store the coincidence index:
      tmp1 = np.repeat(k.upper(), ci_post_tmp.shape[0])  
      tmp2 = np.repeat(('DAP ' + dap_group[i] + '*'), ci_post_tmp.shape[0])
      tmp = pd.DataFrame({'post': ci_post_tmp.ci.values, 'model': tmp1, 'dap': tmp2})   
      ci_post = pd.concat([ci_post, tmp], axis=0)
    print('Model: {}, DAP_i: {}'.format(k, dap_group[i]))

# Store into a list different DAP values interv
dap_group = ['30~45', '30~60', '30~75', '30~90', '30~105']

# Compute coincidence index for the Dynamic Bayesian network:
for i in range(len(dap_group)):
  # Subset expectations and observations for coincidence index:
  expect_tmp = expect_cv2['dbn_cv2_height_trained!on!dap:' + dap_group[i]]
  y_obs_tmp = df[df.trait=='drymass'].y_hat
  # Get the upper bound of the interval:
  upper = int(dap_group[i].split('~')[1])
  # Get the number of selected individuals for 20% selection intensity:
  n_selected = int(y_obs_tmp.size * 0.2)
  # Build the indexes for computing the coincidence index:
  rank_obs = np.argsort(y_obs_tmp)[::-1][0:n_selected]
  # Vector for coincidence indexes:
  ci_post_tmp = pd.DataFrame(index=expect_tmp.index, columns=['ci'])
  # Compute the coincidence index:
  for sim in range(ci_post_tmp.shape[0]):
    # Build the indexes for computing the coincidence index:
    rank_pred = np.argsort(expect_tmp.iloc[sim])[::-1][0:n_selected]
    # Selected lines:
    top_lines_pred = df.id_gbs[expect_tmp.iloc[sim].iloc[rank_pred].index]
    # Compute coincidence index in the top 20% or not: 
    ci_post_tmp.iloc[sim] = top_lines_pred.isin(top_lines_obs).mean(axis=0)
  # Store the coincidence index:
  tmp1 = np.repeat('DBN', ci_post_tmp.shape[0])  
  tmp2 = np.repeat(('DAP ' + str(upper) + '*'), ci_post_tmp.shape[0])
  tmp = pd.DataFrame({'post': ci_post_tmp.ci.values, 'model': tmp1, 'dap': tmp2})   
  ci_post = pd.concat([ci_post, tmp], axis=0)
  print('Model: dbn, DAP_i: {}'.format(dap_group[i]))

# Changing data type for the post column:
ci_post['post'] = ci_post['post'].astype(float)

# Changing labels:
ci_post.columns = ['Days after planting', 'Model', 'Coincidence index posterior values']

# Set directory:
os.chdir(prefix_proj + 'plots/cv/ciplot')

# Plotting coincidence indexes:
ax = sns.violinplot(x='Days after planting',
                    y='Coincidence index posterior values',
                    data=ci_post,
                    hue='Model')
plt.ylim(0.12, 0.42)
plt.savefig("ci_plot.pdf", dpi=150)
plt.savefig("ci_plot.png", dpi=150)
plt.clf()


#-------------------------------Inspection of the bin probability relevance----------------------------------#

# # Set the directory:
# os.chdir(prefix_out + 'data')

# # Read the data frame with the loci mapping information:
# loci_info = pd.read_csv("loci_info.csv", index_col=0)

# # Set the directory:
# os.chdir(prefix_out + 'outputs/cross_validation/DBN/cv2-30~105/height/')

# # Load stan fit object and model:
# with open("output_dbn-0~5.pkl", "rb") as f:
#   data_dict = pickle.load(f)

# # Index the fit object and model
# out = data_dict['fit'].extract()

# # Define a small interval around zero to compute probabilities:
# lamb = 0.2

# # Compute indicators with the values are larger then the intervals centered in zero in both directions:
# ind_bin = (out['alpha_1'] > lamb) | (out['alpha_1'] < -1*lamb) 

# # Compute the probability the effects are larger then the intervals centered in zero in both directions:
# prob_bin = pd.DataFrame({'prob': ind_bin.mean(axis=0), 'bin': loci_info.bin.unique()})

# # Generate the probabilistic Manhattan plot:
# sns.barplot(x=prob_bin.bin, y=prob_bin.prob)

# # Display plot:
# plt.show()

# # Selecting five bins under largest probabilities:
# bin_cand = prob_bin.sort_values(by=['prob'], ascending=False).iloc[0:4]

# # Getting the start and end position of the bin displaying the first largest probability:
# tmp = loci_info[loci_info.bin == bin_cand.bin.iloc[2]]

# pos = tmp.iloc[[0, tmp.shape[0]-1],:]

# ##########

# # Creating a column to receive the chromossomes:
# prob_bin['chrom'] = np.nan

# # Adding the values to the previous created column:
# for i in prob_bin.bin:
#   mask = loci_info.bin==i
#   prob_bin.chrom[prob_bin.bin == i] = loci_info.chrom[mask].unique()[0]
#   print(i)


# prob_bin[['bin', 'chrom']].drop_duplicates()

# loci_info[['prob', 'chrom']].plot(kind='bar', colormap='Reds')

# loci_info.sort_values(by=['chrom'])[['chrom', 'bin']].drop_duplicates()


#----------------------------------------------Restore data--------------------------------------------------#

# Load libraries:
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
import re
import pystan as ps
from scipy.stats.stats import pearsonr
from pprint import pprint as pprint
from pymc3.stats import hpd 
import argparse
parser = argparse.ArgumentParser()


# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs was saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# # Saving data:
# os.chdir(prefix_out + "outputs/tmp")
# data = [y, X, y_pred_cv1, y_pred_cv2, y_obs_cv1, y_obs_cv2, prob_dict, expect_cv2, df]
# np.savez('mtrait_results.npz', data)

# Loading data:
os.chdir(prefix_out + "outputs/tmp")
container = np.load('mtrait_results.npz')
data = [container[key] for key in container]
y = data[0][0]
X = data[0][1]
y_pred_cv1 = data[0][2]
y_pred_cv2 = data[0][3]
y_obs_cv1 = data[0][4]
y_obs_cv2 = data[0][5]
prob_dict = data[0][6]
expect_cv2 = data[0][7]
df = data[0][8]



#--------------------------------------------For Latter usage------------------------------------------------#


# prob=prob_dict["bn_45_120"]

# n_for_plot = 25

# upper_slice = np.argsort(prob).iloc[0:n_for_plot]
# lower_slice = np.argsort(prob).iloc[(prob.size - n_for_plot):(prob.size)]

# slice_prob = pd.concat([upper_slice,lower_slice], axis=0) 

# tmp = pd.DataFrame({'Sorghum inbred lines': df.id_gbs[prob.index].iloc[slice_prob], 'Probabilities': prob.iloc[slice_prob].values}) 

# tmp = pd.DataFrame({'Sorghum inbred lines': df.id_gbs[prob.index], 'Probabilities': prob.values}) 

# p1 = sns.barplot(x='Probabilities', y='Sorghum inbred lines', data=tmp)
# p1.set(yticklabels=[])
# plt.xlim(0, 1)
# plt.show()


