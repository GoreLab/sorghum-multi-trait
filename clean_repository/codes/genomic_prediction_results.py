
#------------------------------------------------Modules-----------------------------------------------------#

# Load libraries:
import pandas as pd
import numpy as np
import os
import pickle
import argparse
from scipy.stats.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
parser = argparse.ArgumentParser()

# Turn off interactive mode:
plt.ioff()


#-------------------------------------------Add flags to the code--------------------------------------------#

# Get flags:
parser.add_argument("-rpath", "--rpath", dest = "rpath", help="The path of the repository")
parser.add_argument("-opath", "--opath", dest = "opath", help="The path of the folder with general outputs")

# Parse the paths:
args = parser.parse_args()

# Subset arguments:
REPO_PATH = args.rpath
OUT_PATH = args.opath
# REPO_PATH = '/workdir/jp2476/sorghum-multi-trait'
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'


#-----------------------------------------Read train and test data-------------------------------------------#

# Type of forward-chaining cross-validation (fcv) schemes:
fcv_type = ['fcv-30~45', 'fcv-30~60', 'fcv-30~75', 'fcv-30~90', 'fcv-30~105',
      'fcv-30~only', 'fcv-45~only', 'fcv-60~only', 'fcv-75~only', 'fcv-90~only', 'fcv-105~only']

# Create a list of the 5-fold cross-validation (cv5f) folds:
cv5f_fold = ['k0', 'k1', 'k2', 'k3', 'k4']

# Create a list with the traits set:
trait_set = ['drymass', 'height']

# Initialize list to receive the outputs:
y = dict()
X = dict()

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Read cv5f files:
for s in range(len(trait_set)):
  for j in range(len(cv5f_fold)):
    # Create the suffix of the file name for the cv5f data case:
    index = 'cv5f_' + trait_set[s] + '_' + cv5f_fold[j] + '_tst'
    # Read data:
    y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
    X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)

# Read fcv files for height:
for t in range(len(fcv_type)):
  for j in range(len(cv5f_fold)):
    # Create the suffix of the file name for the fcv data case:
    index = fcv_type[t] + '_height_tst'
    # Read data:
    y[index] = pd.read_csv('y_' + index + '.csv', header = 0, index_col=0)
    X[index] = pd.read_csv('x_' + index + '.csv', header = 0, index_col=0)


#----------------------------Organize data into a new format for easy subsetting-----------------------------#

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)
df.dap = df.dap.fillna(0).astype(int)

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Initialize dictionary to receive observations for cv5f analysis:
y_obs_cv5f = dict()

# Store observations for height stratified by modelling scenario for cv5f analysis:
for d in dap_group:
  y_obs_cv5f['cv5f_height_dap:' + d] = df.y_hat[(df.trait=='height') & (df.dap==int(d))] 

# Store observations for drymass stratified by modelling scenario for cv5f analysis:
y_obs_cv5f['cv5f_drymass'] = df.y_hat[df.trait=='drymass'] 

# Initialize dictionary to receive observations for fcv analysis:
y_obs_fcv = dict()

# Store observations for height stratified by modelling scenario for fcv analysis:
dap_group = ['30', '45', '60', '75', '90', '105']  
for d in dap_group:
  y_obs_fcv['fcv_height_for!trained!on:' + d] = y['fcv-' + d + '~only_height_tst']

# Store observations for height stratified by modelling scenario for fcv analysis:
dap_group = ['30~45', '30~60', '30~75', '30~90', '30~105']
for d in dap_group:
  y_obs_fcv['fcv_height_for!trained!on:' + d] = y['fcv-' + d + '_height_tst']


#--------------------------------Compute predictions for the cv5f scheme-------------------------------------#

# Initialize list to receive the predictions:
y_pred_cv5f = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Compute predictions for the BN model:
for s in trait_set:
  # Compute predictions for drymass:
  if s=='drymass':
    for j in cv5f_fold:
      # Set the directory:
      os.chdir(OUT_PATH + '/cv/BN/cv5f/drymass/' + j)
      # Load stan fit object and model:
      with open("output_bn_fit_0.pkl", "rb") as f:
          data_dict = pickle.load(f)
      # Index the fit object and model
      out = data_dict['fit'].extract()
      # Index and subsetting the feature matrix:
      index1 = 'cv5f_drymass_' + j + '_tst'
      index2 = 'bn_cv5f_drymass'
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
        y_pred_cv5f[index2] = tmp
      if j!='k0':
        y_pred_cv5f[index2] = pd.concat([y_pred_cv5f[index2], tmp], axis=1)
  # Compute predictions for height:
  if s=='height':
    for d in range(len(dap_group)):
      for j in cv5f_fold:
        # Set the directory:
        os.chdir(OUT_PATH + '/cv/BN/cv5f/height/' + j)
        # Load stan fit object and model:
        with open("output_bn_fit_" + str(d) + ".pkl", "rb") as f:
            data_dict = pickle.load(f)
        # Index the fit object and model
        out = data_dict['fit'].extract()
        # Index and subsetting the feature matrix:
        index1 = 'cv5f_height_' + j + '_tst'
        index2 = 'bn_cv5f_height_trained!on!dap:' + dap_group[d]
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
          y_pred_cv5f[index2] = tmp
        if j!='k0':
          y_pred_cv5f[index2] = pd.concat([y_pred_cv5f[index2], tmp], axis=1)

# Compute predictions for the PBN model:
for d in range(len(dap_group)):
  for j in cv5f_fold:
    # Set the directory:
    os.chdir(OUT_PATH + '/cv/PBN/cv5f/drymass-height/' + j)
    # Load stan fit object and model:
    with open("output_pbn_fit_" + str(d) + ".pkl", "rb") as f:
        data_dict = pickle.load(f)
    # Index the fit object and model
    out = data_dict['fit'].extract()
    # Index and subset the feature matrix:
    index1_0 = 'cv5f_drymass_' + j + '_tst'
    index1_1 = 'cv5f_height_' + j + '_tst'
    index2_0 = 'pbn_cv5f_drymass_trained!on!dap:' + dap_group[d]
    index2_1 = 'pbn_cv5f_height_trained!on!dap:' + dap_group[d]
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
      y_pred_cv5f[index2_0] = tmp_0
      y_pred_cv5f[index2_1] = tmp_1
    if j!='k0':
      y_pred_cv5f[index2_0] = pd.concat([y_pred_cv5f[index2_0], tmp_0], axis=1)
      y_pred_cv5f[index2_1] = pd.concat([y_pred_cv5f[index2_1], tmp_1], axis=1)

# List drymass predictions for ensambling:
tmp = []
for i in range(len(dap_group)):
  tmp.append(y_pred_cv5f['pbn_cv5f_drymass_trained!on!dap:' + dap_group[i]])

# Ensamble predictions for drymass:
y_pred_cv5f['pbn_cv5f_drymass_ensambled'] = (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]) / 7

# Remove predictions that will not be used anymore:
for d in range(len(dap_group)):
  y_pred_cv5f.pop('pbn_cv5f_drymass_trained!on!dap:' + dap_group[d])

# Compute predictions for the DBN model:
for j in cv5f_fold:
  # Set the directory:
  os.chdir(OUT_PATH + '/cv/DBN/cv5f/height/' + j)
  # Load stan fit object and model:
  with open("output_dbn-0~6.pkl", "rb") as f:
      data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  for d in range(len(dap_group)):
    # Index and subset the feature matrix:
    index1 = 'cv5f_height_' + j + '_tst'
    index2 = 'dbn_cv5f_height_trained!on!dap:' + dap_group[d]
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
      y_pred_cv5f[index2] = tmp
    if j!='k0':
      y_pred_cv5f[index2] = pd.concat([y_pred_cv5f[index2], tmp], axis=1)


#--------------------------------Computing predictions for the fcv scheme------------------------------------#

# Initialize list to receive the predictions:
y_pred_fcv = dict()

# Initialize list to receive the expectaions:
expect_fcv = dict()

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105']  

# Compute predictions for the BN model:
for d in range(len(dap_group)):
  # Set the directory:
  os.chdir(OUT_PATH + '/cv/BN/fcv-' + dap_group[d] + '~only/height/')
  # Load stan fit object and model:
  with open("output_bn_fit_0.pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  # Index and subset the feature matrix:
  index1 = 'fcv-' + dap_group[d] + '~only_height_tst'
  index2 = 'bn_fcv_height_trained!on!dap:' + dap_group[d]
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
  y_pred_fcv[index2] = tmp
  # Store expectations:
  expect_fcv[index2] = pd.DataFrame(out['expectation'], index=index, columns=df.index[df.dap==int(dap_group[d])])

# Compute predictions for the PBN model:
for d in range(len(dap_group)):
  # Set the directory:
  os.chdir(OUT_PATH + '/cv/PBN/fcv-' + dap_group[d] + '~only/drymass-height/')
  # Load stan fit object and model:
  with open("output_pbn_fit_0.pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fitted object and model
  out = data_dict['fit'].extract()
  # Index and subset the feature matrix:
  index1 = 'fcv-' + dap_group[d] + '~only_height_tst'
  index2 = 'pbn_fcv_height_trained!on!dap:' + dap_group[d]
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
  y_pred_fcv[index2] = tmp
  # Store expectations:
  expect_fcv[index2] = pd.DataFrame(out['expectation_1'], index=index, columns=df.index[df.dap==int(dap_group[d])])

# Different DAP measures:
fcv_type = ['fcv-30~45', 'fcv-30~60', 'fcv-30~75', 'fcv-30~90', 'fcv-30~105']
dap_index = ['0~1', '0~2', '0~3', '0~4', '0~5']

# Compute predictions for the DBN model:
for c in range(len(fcv_type)):
  # Set the directory:
  os.chdir(OUT_PATH + '/cv/DBN/' + fcv_type[c] + '/height/')
  # Load stan fit object and model:
  with open("output_dbn-" + dap_index[c] + ".pkl", "rb") as f:
    data_dict = pickle.load(f)
  # Index the fit object and model
  out = data_dict['fit'].extract()
  # Get the last time point used for training:
  upper = dap_index[c].split('~')[1]
  # Index and subset the feature matrix:
  index1 = fcv_type[c] +'_height_tst'
  index2 = 'dbn_fcv_height_trained!on!dap:' + fcv_type[c].split('-')[1]
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
  y_pred_fcv[index2] = tmp
  # Store expectations:
  expect_fcv[index2] = pd.DataFrame(out['expectation_'  + upper], index=index, columns=df.index[df.dap==int(dap_group[int(upper)])])


#--------------------Compute prediction accuracies for the cv5f scheme (table and figures)--------------------#

# Different models:
models = ['BN', 'PBN', 'DBN', 'MTi-GBLUP', 'MTr-GBLUP']

# Different DAP measures:
dap_group = ['30', '45', '60', '75', '90', '105', '120']  

# Number of Monte Carlo simulations:
n_sim = 800

# Initialize table to store prediction accuracy based on the posterior mean of the predictions from the Bayesian models:
cv5f_table_mean = pd.DataFrame(index = ['DB'] + ['PH_' + str(i) for i in dap_group], columns=models)

# Models for computing predictive accuracy:
models = ['BN', 'PBN', 'DBN']

# Accuracies based on the posterior means of the predictions for cv5f related to the height prediction:
for m in models:
  for d in dap_group:
    index = y_pred_cv5f[m.lower() + '_cv5f_height_trained!on!dap:' + d].columns
    cor_tmp = np.round(pearsonr(y_pred_cv5f[m.lower() + '_cv5f_height_trained!on!dap:' + d].mean(axis=0), y_obs_cv5f['cv5f_height_dap:' + d][index])[0],2)
    cv5f_table_mean.loc['PH_' + str(d), m] = cor_tmp

# Prediction accuracies based on the posterior means of the predictions for cv5f related to dry biomass prediction:
index = y_pred_cv5f['bn_cv5f_drymass'].columns
cv5f_table_mean.loc['DB', 'BN'] = np.round(pearsonr(y_pred_cv5f['bn_cv5f_drymass'].mean(axis=0), y_obs_cv5f['cv5f_drymass'][index])[0],2)

index = y_pred_cv5f['pbn_cv5f_drymass_ensambled'].columns
cv5f_table_mean.loc['DB', 'PBN'] = np.round(pearsonr(y_pred_cv5f['pbn_cv5f_drymass_ensambled'].mean(axis=0), y_obs_cv5f['cv5f_drymass'][index])[0],2)

# Initialize table to store the standard deviation of the prediction accuracies based on the samples of the posterior distribution of the predictions obtained by the Bayesian models:
cv5f_table_std = pd.DataFrame(index = ['DB'] + ['PH_' + str(i) for i in dap_group], columns=models)

# Standard deviation of the prediction accuracies based on the samples of the posterior distribution of the predictions obtained by the Bayesian models for height:
for m in models:
  # Vector to store the predictive accuracy of each posterior draw:
  cor_tmp = pd.DataFrame(index=["s_" + str(i) for i in range(n_sim)], columns=['acc'])
  # To compute the standard deviation of the accuracies in the posterior level:
  for d in dap_group:
    index1 = y_pred_cv5f[m.lower() + '_cv5f_height_trained!on!dap:' + d].columns
    index2 = y_pred_cv5f[m.lower() + '_cv5f_height_trained!on!dap:' + d].index
    for s in index2:
      cor_tmp.loc[s] = pearsonr(y_pred_cv5f[m.lower() + '_cv5f_height_trained!on!dap:' + d].loc[s], y_obs_cv5f['cv5f_height_dap:' + d][index1])[0]
      print('Model: {}, DAP: {}, Simulation: {}'.format(m.upper(), d, s))
    # Printing the standard deviation of the predictive accuracies:
    cv5f_table_std.loc['PH_' + str(d), m] = float(np.round(cor_tmp.std(axis=0),3))

# Different models:
models = ['bn_biomass', 'pbn_biomass']

# Standard deviation of the prediction accuracies based on the samples of the posterior distribution of the predictions obtained by the Bayesian models for dry biomass:
for m in models:
  # Vector to store the predictive accuracy of each posterior draw:
  cor_tmp = pd.DataFrame(index=["s_" + str(i) for i in range(n_sim)], columns=['acc'])
  # To compute the standard deviation of the accuracies in the posterior level:
  for d in dap_group:
    if m=='bn_biomass':
      index1 = y_pred_cv5f['bn_cv5f_drymass'].columns
      index2 = y_pred_cv5f['bn_cv5f_drymass'].index
    if m=='pbn_biomass':
      index1 = y_pred_cv5f['pbn_cv5f_drymass_ensambled'].columns
      index2 = y_pred_cv5f['pbn_cv5f_drymass_ensambled'].index
    for s in index2:
      if m=='bn_biomass':
        cor_tmp.loc[s] = pearsonr(y_pred_cv5f['bn_cv5f_drymass'].loc[s], y_obs_cv5f['cv5f_drymass'][index1])[0]
      if m=='pbn_biomass':
        cor_tmp.loc[s] = pearsonr(y_pred_cv5f['pbn_cv5f_drymass_ensambled'].loc[s], y_obs_cv5f['cv5f_drymass'][index1])[0]
      print('Model: {}, Simulation: {}'.format(m.upper(), s))
    if m=='bn_biomass':
      # Printing the standard deviation of the predictive accuracies:
      cv5f_table_std.loc['DB', 'BN'] = float(np.round(cor_tmp.std(axis=0),3))
    if m=='pbn_biomass':
      # Printing the standard deviation of the predictive accuracies:
      cv5f_table_std.loc['DB', 'PBN'] = float(np.round(cor_tmp.std(axis=0),3))


#-----------------------------Compute prediction accuracies for the fcv scheme--------------------------------#

# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive the accuracy matrices:
cor_dict = dict()

# Compute correlation for the Baysian Network and Pleiotropic Bayesian Network model under fcv scheme:
for k in model_set:
  # Create an empty correlation matrix:
  cor_tmp = np.empty([len(dap_group)]*2)
  cor_tmp[:] = np.nan
  for i in range(len(dap_group[:-1])):
    # Subset predictions for correlation computation:
    y_pred_tmp = y_pred_fcv[k + '_fcv_height_trained!on!dap:' + dap_group[i]].mean(axis=0)
    y_obs_tmp = y_obs_fcv['fcv_height_for!trained!on:' + dap_group[i]].y_hat
    for j in range(len(dap_group)):
      # Conditional to compute correlation just forward in time:
      if (j>i):
        # Subset indexes for subsetting the data:
        subset = df[df.dap==int(dap_group[j])].index
        # Build correlation matrix for the Bayesian Network model:
        cor_tmp[i, j] = np.round(pearsonr(y_pred_tmp[subset], y_obs_tmp[subset])[0],4)
  # Store the computed correlation matrix for the Bayesian network model under fcv scheme:
  cor_dict['fcv_' + k] = cor_tmp

# Store into a list different DAP values intervals:
dap_group1 = ['30~45', '30~60', '30~75', '30~90', '30~105']
dap_group2 = ['30', '45', '60', '75', '90', '105', '120']

# Create an empty correlation matrix:
cor_tmp = np.empty([len(dap_group)]*2)
cor_tmp[:] = np.nan

# Compute correlation for the Dynamic Bayesian Network model under fcv scheme:
for i in range(len(dap_group1)):
  # Subset predictions for correlation computation:
  y_pred_tmp = y_pred_fcv['dbn_fcv_height_trained!on!dap:' + dap_group1[i]].mean(axis=0)
  y_obs_tmp = y_obs_fcv['fcv_height_for!trained!on:' + dap_group1[i]].y_hat
  for j in range(len(dap_group2)):    
    # Get the upper bound of the interval:
    upper = int(dap_group1[i].split('~')[1])
    # Conditional to compute correlation just forward in time:
    if (int(dap_group2[j])>upper):
      # Subset indexes for subsetting the data:
      subset = df[df.dap==int(dap_group2[j])].index
      # Build correlation matrix for the Bayesian Network model:
      cor_tmp[i, j] = np.round(pearsonr(y_pred_tmp[subset], y_obs_tmp[subset])[0],4)

# Store the computed correlation matrix for the Bayesian network model under fcv scheme:
cor_dict['fcv_dbn'] = cor_tmp

# Builda mask just to filter accuracies common across models:
mask = np.isnan(cor_dict['fcv_dbn'])

# Eliminate values not shared across all models:
cor_dict['fcv_bn'][mask] = np.nan
cor_dict['fcv_pbn'][mask] = np.nan

# Eliminate rows and columns without any correlation value:
cor_dict['fcv_bn'] = cor_dict['fcv_bn'][0:5,2:7]
cor_dict['fcv_pbn'] = cor_dict['fcv_pbn'][0:5,2:7]
cor_dict['fcv_dbn'] = cor_dict['fcv_dbn'][0:5,2:7]

# Print predictive accuracies
print(cor_dict['fcv_bn'])
print(cor_dict['fcv_pbn'])
print(cor_dict['fcv_dbn'])

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

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
  heat = sns.heatmap(np.flip(np.flip(cor_dict['fcv_' + i],axis=1), axis=0),
             linewidths=0.25,
             cmap='YlOrBr',
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
  plt.savefig("heatplot_fcv_" + i + "_accuracy.pdf", dpi=350)
  plt.savefig("heatplot_fcv_" + i + "_accuracy.png", dpi=350)
  plt.clf()


#------------------------------Compute coincidence index based on lines (CIL)---------------------------------#

# Dictionary to receive CIL:
cil_dict = dict()

# Store into a list different DAP values intervals:
dap_group1 = ['30~45', '30~60', '30~75', '30~90', '30~105']
dap_group2 = ['30', '45', '60', '75', '90', '105', '120']

# Compute CIL for the Dynamic Bayesian network model (best Bayesian model on the fcv):
for i in range(len(dap_group1)):
	# Subset predictions for correlation computation:
	y_pred_tmp = y_pred_fcv['dbn_fcv_height_trained!on!dap:' + dap_group1[i]]
	y_obs_tmp = y_obs_fcv['fcv_height_for!trained!on:' + dap_group1[i]].y_hat
	for j in range(len(dap_group2)):
		# Get the upper bound of the interval:
		upper = int(dap_group1[i].split('~')[1])
		# Conditional to compute correlation just forward in time:
		if (int(dap_group2[j])>upper):
			# Subset indexes for subsetting the data:
			subset = df[df.dap==int(dap_group2[j])].index
			# Get the number of selected individuals for 20% selection intensity:
			n_selected = int(y_obs_tmp[subset].size * 0.2)
			# Build the indexes for computing the CIL:
			top_rank_obs = np.argsort(y_obs_tmp[subset])[::-1][0:n_selected]
			# Vector for storing the indicators:
			ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
			# Build CIL matrix for the Bayesian Network model:
			for sim in range(y_pred_tmp.shape[0]):
				# Top predicted lines:
				top_rank_pred = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1][0:n_selected]
				# Top indicator:
				ind_tmp = top_rank_pred.isin(top_rank_obs)
				ind_vec.loc['s_' + str(sim), ind_tmp.index] = ind_tmp
			# Index to store CIL into dictionary:
			index = 'dbn_' + dap_group1[i] + '_' + dap_group2[j]
			# Compute CIL:
			cil_dict[index]=ind_vec.mean(axis=0)
			print('Model: dbn, DAP_i: {}, DAP_j: {}'.format(dap_group1[i], dap_group2[j]))

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

# DAP groups used for plotting
dap_group = ['45', '60', '75', '90', '105']

# Generating panel plot:
for i in range(len(dap_group)):
	# Subset CIL for plotting:
	cil=cil_dict['dbn_30~' + dap_group[i] + '_120']
	# Remove entries from other time points with nan:
	cil = cil[~np.isnan(cil)]
	# Get the order of the first slice used to train, to plot the order of the other slices:
	if i==0:
		# Get the order decreasing of the CIL:
		order_index = np.argsort(cil)[::-1]
		# Individuais displaying CIL higher then 80%:
		mask = cil.iloc[order_index] > 0.5
	# Subset CIL for plotting:
	p1 = cil.iloc[order_index][mask].plot.barh(color='red', figsize=(10,12))
	p1.set(yticklabels=df.id_gbs[mask.index])
	p1.tick_params(axis='y', labelsize=5)
	p1.tick_params(axis='x', labelsize=12)
	plt.xlabel('Top 20% posterior coincidence index based on lines', fontsize=20)
	plt.ylabel('Sorghum lines', fontsize=20)
	plt.xlim(0.5, 1)
	plt.savefig('cilplot_fcv_' + 'dbn_30~' + dap_group[i] + '_120.pdf', dpi=350)
	plt.savefig('cilplot_fcv_' + 'dbn_30~' + dap_group[i] + '_120.png', dpi=350)
	plt.clf()


#----Compute coincidence index for dry biomass indirect selection using height adjusted means time series-----#

# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Compute coincidence index for the Bayesian network and Pleiotropic Bayesian Network model:
for k in model_set:
  for i in range(len(dap_group[:-1])):
  	# Subset expectations and observations for coincidence index computation:
  	expect_tmp = expect_fcv[k + '_fcv_height_trained!on!dap:' + dap_group[i]].copy(deep=True)
  	# Get the name of the lines from the entries of the expectation:
  	expect_tmp.columns = df.loc[expect_tmp.columns].id_gbs
  	# Subset data frame:
  	df_tmp = df[df.trait=='drymass']
  	df_tmp.index = df_tmp.id_gbs
  	# Subset observations:
  	y_obs_tmp = df_tmp.y_hat[expect_tmp.columns]
  	# Get the number of selected individuals for 20% selection intensity:
  	n_selected = int(y_obs_tmp.size * 0.2)
  	# Build the indexes for computing the coincidence index:
  	top_rank_obs = np.argsort(y_obs_tmp)[::-1][0:n_selected]
  	# Vector for coincidence indexes:
  	ci_post_tmp = pd.DataFrame(index=expect_tmp.index, columns=['ci'])
  	# Compute the coincidence index:
  	for sim in range(ci_post_tmp.shape[0]):
  		# Build the indexes for computing the coincidence index:
  		top_rank_pred = np.argsort(expect_tmp.loc['s_' + str(sim)])[::-1][0:n_selected]
  		# Compute coincidence index in the top 20% or not: 
  		ci_post_tmp.loc['s_' + str(sim)] = top_rank_pred.isin(top_rank_obs).mean(axis=0)
  	if (k==model_set[0]) & (i==0):
  	  # Store the coincidence index:
  	  ci_post = pd.DataFrame(columns=['post', 'model', 'dap'])
  	  ci_post['model'] = np.repeat(k.upper(), ci_post_tmp.shape[0])
  	  ci_post['dap'] = np.repeat((dap_group[i] + '*'), ci_post_tmp.shape[0])
  	  ci_post['post'] = ci_post_tmp.ci.values
  	else:
  	  # Store the coincidence index:
  	  tmp1 = np.repeat(k.upper(), ci_post_tmp.shape[0])  
  	  tmp2 = np.repeat((dap_group[i] + '*'), ci_post_tmp.shape[0])
  	  tmp = pd.DataFrame({'post': ci_post_tmp.ci.values, 'model': tmp1, 'dap': tmp2})   
  	  ci_post = pd.concat([ci_post, tmp], axis=0)
  	print('Model: {}, DAP_i: {}'.format(k, dap_group[i]))

# Store into a list different DAP values interv
dap_group = ['30~45', '30~60', '30~75', '30~90', '30~105']

# Compute coincidence index for the Dynamic Bayesian network:
for i in range(len(dap_group)):
	# Subset expectations:
	expect_tmp = expect_fcv['dbn_fcv_height_trained!on!dap:' + dap_group[i]].copy(deep=True)
	# Get the name of the lines from the entries of the expectation:
	expect_tmp.columns = df.loc[expect_tmp.columns].id_gbs
	# Sub set data frame:
	df_tmp = df[df.trait=='drymass']
	df_tmp.index = df_tmp.id_gbs
	# Subset observations:
	y_obs_tmp = df_tmp.y_hat[expect_tmp.columns]
	# Get the upper bound of the interval:
	upper = int(dap_group[i].split('~')[1])
	# Get the number of selected individuals for 20% selection intensity:
	n_selected = int(y_obs_tmp.size * 0.2)
	# Build the indexes for computing the coincidence index:
	top_rank_obs = np.argsort(y_obs_tmp)[::-1][0:n_selected]
	# Vector for coincidence indexes:
	ci_post_tmp = pd.DataFrame(index=expect_tmp.index, columns=['ci'])
	# Compute the coincidence index:
	for sim in range(ci_post_tmp.shape[0]):
		# Build the indexes for computing the coincidence index:
		top_rank_pred = np.argsort(expect_tmp.loc['s_' + str(sim)])[::-1][0:n_selected]
		# Compute coincidence index in the top 20% or not: 
		ci_post_tmp.iloc[sim] = top_rank_pred.isin(top_rank_obs).mean(axis=0)
	# Store the coincidence index:
	tmp1 = np.repeat('DBN', ci_post_tmp.shape[0])  
	tmp2 = np.repeat((str(upper) + '*'), ci_post_tmp.shape[0])
	tmp = pd.DataFrame({'post': ci_post_tmp.ci.values, 'model': tmp1, 'dap': tmp2})   
	ci_post = pd.concat([ci_post, tmp], axis=0)
	print('Model: dbn, DAP_i: {}'.format(dap_group[i]))

# Change data type:
ci_post['post'] = ci_post['post'].astype(float)

# Change labels:
ci_post.columns = ['Coincidence index posterior values', 'Model', 'Days after planting']

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

# Plot coincidence indexes:
plt.figure(figsize=(20,15))
with sns.plotting_context(font_scale=1):
	ax = sns.violinplot(x='Days after planting',
	                    y='Coincidence index posterior values',
	                    data=ci_post,
	                    hue='Model')
	plt.ylim(0.12, 0.4)
	ax.tick_params(labelsize=30)
	plt.xlabel('Days after planting', fontsize=40)
	plt.ylabel(	'Coincidence index posterior values', fontsize=40)
	plt.legend(fontsize='xx-large', title_fontsize=40)
	plt.savefig("ci_plot.pdf", dpi=350)
	plt.savefig("ci_plot.png", dpi=350)
	plt.clf()


#--------------------------Accuracy heatmap for the multivariate linear mixed model--------------------------#

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/tables")

# Initialize a dictionary to receive the accuracies:
gblup_dict = dict()

# Load predictive accuracies of the GBLUP models and store on the final table:
gblup_dict['MTi-GBLUP_cv5f'] = pd.read_csv('acc_MTi-GBLUP_cv5f.csv', header = 0, index_col=0)
gblup_dict['MTr-GBLUP_cv5f'] = pd.read_csv('acc_MTr-GBLUP_cv5f.csv', header = 0, index_col=0)
cv5f_table_mean['MTi-GBLUP'][gblup_dict['MTi-GBLUP_cv5f'].index] = np.round(gblup_dict['MTi-GBLUP_cv5f'].values.flatten(),3)
cv5f_table_mean['MTr-GBLUP'][gblup_dict['MTr-GBLUP_cv5f'].index] = np.round(gblup_dict['MTr-GBLUP_cv5f'].values.flatten(),3)

# Load correlation matrices:
gblup_dict['MTi-GBLUP_fcv'] = pd.read_csv('acc_MTi-GBLUP_fcv.csv', header = 0, index_col=0)
gblup_dict['MTr-GBLUP_fcv'] = pd.read_csv('acc_MTr-GBLUP_fcv.csv', header = 0, index_col=0)

# List of models to use:
model_set = ['MTi-GBLUP', 'MTr-GBLUP']

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

# Generate accuracy heatmaps:
for i in range(len(model_set)):
  # Labels for plotting the heatmap for the Dynamic Bayesian model:
  labels_axis0 = ['DAP 30:45*', 'DAP 30:60*', 'DAP 30:75*', 'DAP 30:90*', 'DAP 30:105*']
  # Labels for plotting the heatmap:
  labels_axis1 = ['DAP 60', 'DAP 75', 'DAP 90', 'DAP 105', 'DAP 120']
  # Heat map of the adjusted means across traits:
  heat = sns.heatmap(gblup_dict[model_set[i] + '_fcv'],
             linewidths=0.25,
             cmap='YlOrBr',
             vmin=0.3,
             vmax=1,
             annot=True,
             annot_kws={"size": 18},
             xticklabels=labels_axis0,
             yticklabels=labels_axis1)
  heat.set_ylabel('')    
  heat.set_xlabel('')
  plt.xticks(rotation=25)
  plt.yticks(rotation=45)
  plt.savefig("heatplot_fcv_" + model_set[i] + "_accuracy.pdf", dpi=350)
  plt.savefig("heatplot_fcv_" + model_set[i] + "_accuracy.png", dpi=350)
  plt.clf()

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/tables")

# Save tables with the results from the cv5f:
cv5f_table_mean.to_csv("cv5f_accuracy_table_mean_all_genomic_prediction_models.csv")
cv5f_table_std.to_csv("cv5f_accuracy_table_std_bayesian_genomic_prediction_models.csv")


###############################################################

# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'

# # Set the directory to store processed data:
# os.chdir(OUT_PATH + "/processed_data")

# data = [y, X, y_pred_cv5f, y_pred_fcv, y_obs_cv5f, y_obs_fcv, cil_dict, expect_fcv, df]
# np.savez('mtrait_results.npz', data)

# # Loading data:
# os.chdir(OUT_PATH + "/processed_data")
# container = np.load('mtrait_results.npz', allow_pickle=True)
# data = [container[key] for key in container]
# y = data[0][0]
# X = data[0][1]
# y_pred_cv5f = data[0][2]
# y_pred_fcv = data[0][3]
# y_obs_cv5f = data[0][4]
# y_obs_fcv = data[0][5]
# cil_dict = data[0][6]
# expect_fcv = data[0][7]
# df = data[0][8]


