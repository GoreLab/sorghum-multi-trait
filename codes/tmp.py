

j='k1'

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

# Creating the indexes for creating a data frame to receive predictions:
id_gbs = df.id_gbs[X_tmp.index]
dap = df.dap[X_tmp.index]
index = ["s_" + str(i) for i in range(out['mu'].size)]
col_name = [(str(id_gbs.iloc[i]) + '_' + str(dap.iloc[i])) for i in range(X_tmp.shape[0])]

# Initialize a matrix to receive the posterior predictions:
tmp = pd.DataFrame(index=index, columns=col_name)

# Computing predictions:
for sim in range(out['mu'].size):
  # Subsetting parameters:
  mu = out['mu'][sim]
  alpha = out['alpha'][sim,:]
  # Prediction:
  tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values

# Store prediction:
if j=='k0':
  y_pred_cv1[index2] = tmp

if j!='k0':
  y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=1)


####### 

d = 0
j = cv1_fold[1]

# Set the directory:
os.chdir(prefix_out + 'outputs/cross_validation/BN/cv1/height/' + j)
# Loading stan fit object and model:
with open("output_bn_fit_" + str(d) + ".pkl", "rb") as f:
    data_dict = pickle.load(f)
# Index the fit object and model
out = data_dict['fit'].extract()
# Index and subsetting the feature matrix:
index1 = 'cv1_height_' + j + '_tst'
index2 = 'bn_cv1_height_trained!on!dap:' + dap_group[d]
X_tmp = X[index1][X[index1].iloc[:,0] == int(dap_group[d])]
X_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
# Creating the indexes for creating a data frame to receive predictions:
id_gbs = df.id_gbs[X_tmp.index]
dap = df.dap[X_tmp.index]
index = ["s_" + str(i) for i in range(out['mu'].size)]
col_name = [(str(id_gbs.iloc[i]) + '_' + str(dap.iloc[i])) for i in range(X_tmp.shape[0])]
# Initialize a matrix to receive the posterior predictions:
tmp = pd.DataFrame(index=index, columns=col_name)
# Computing predictions:
for sim in range(out['mu'].size):
  # Subsetting parameters:
  mu = out['mu'][sim]
  alpha = out['alpha'][sim,:]
  # Prediction:
  tmp.iloc[sim] = (mu + X_tmp.dot(alpha)).values
# Store prediction:
if j=='k0':
  y_pred_cv1[index2] = tmp
if j!='k0':
  y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=1)

######

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
# Creating the indexes for creating a data frame to receive predictions:
id_gbs = df.id_gbs[X_tmp_0.index]
dap = df.dap[X_tmp_0.index]
index = ["s_" + str(i) for i in range(out['mu_0'].size)]
col_name = [(str(id_gbs.iloc[i]) + '_' + str(dap.iloc[i])) for i in range(X_tmp_0.shape[0])]
# Initialize a matrix to receive the posterior predictions:
tmp_0 = pd.DataFrame(index=index, columns=col_name)
# Creating the indexes for creating a data frame to receive predictions:
id_gbs = df.id_gbs[X_tmp_1.index]
dap = df.dap[X_tmp_1.index]
index = ["s_" + str(i) for i in range(out['mu_0'].size)]
col_name = [(str(id_gbs.iloc[i]) + '_' + str(dap.iloc[i])) for i in range(X_tmp_1.shape[0])]
# Initialize a matrix to receive the posterior predictions:
tmp_1 = pd.DataFrame(index=index, columns=col_name)
# Computing predictions:
for sim in range(out['mu_0'].size):
  # Subsetting parameters:
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

######

for j in cv1_fold:

j='k0'

# Setting the directory:
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
  # Creating the indexes for creating a data frame to receive predictions:
  id_gbs = df.id_gbs[X_tmp.index]
  dap = df.dap[X_tmp.index]
  index = ["s_" + str(i) for i in range(out['mu_' + str(d)].size)]
  col_name = [(str(id_gbs.iloc[i]) + '_' + str(dap.iloc[i])) for i in range(X_tmp.shape[0])]
  # Initialize a matrix to receive the posterior predictions:
  tmp = pd.DataFrame(index=index, columns=col_name)
  # Computing predictions:
  for sim in range(out['mu_' + str(d)].size):
    # Subsetting parameters:
    mu = out['mu_' + str(d)][sim]
    alpha = out['alpha_' + str(d)][sim,:]
    # Prediction:
    tmp.iloc[sim] = (mu + Z_tmp.dot(alpha)).values
  # Store prediction:
  if j=='k0':
    y_pred_cv1[index2] = tmp
  if j!='k0':
    y_pred_cv1[index2] = pd.concat([y_pred_cv1[index2], tmp], axis=0)



