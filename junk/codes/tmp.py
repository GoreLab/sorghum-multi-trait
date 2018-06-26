
# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive the accuracy matrices:
prob_dict = dict()

# Compute correlation for the Baysian network and Pleiotropic Bayesian Network model under CV2 scheme:
for k in model_set:

k='bn'

# Create an empty probability matrix:
prob_tmp = np.empty([len(dap_group)]*2)
prob_tmp[:] = np.nan

for i in range(len(dap_group[:-1])):

i=0


# Subsetting predictions for probability computation:
y_pred_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]

y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group[i]].y_hat

for j in range(len(dap_group)):

j=1

# Conditional to compute probability just forward in time:
if (j>i):

# Subset indexes for subsetting the data:
subset = df[df.dap==int(dap_group[j])].index


ci = np.empty(y_pred_tmp.shape[0])
ci[:] = np.nan

# Build probability matrix for the Bayesian Network model:
for sim in range(y_pred_tmp.shape[0]):
	# Getting the number of selected individuals for 20% selection intensity:
	n_selected = int(y_obs_tmp[subset].size * 0.2)
	# Building the indexes for computing the coeficient of coincidence:
	index_pred = np.argsort(y_pred_tmp.iloc[sim][subset])[::-1][0:n_selected]
	index_obs = np.argsort(y_obs_tmp[subset])[::-1][0:n_selected]
	# Computing the coeficient of coincidence:
	ci[sim] = index_obs.isin(index_pred).sum()/n_selected
	print(sim)




y_obs_tmp[subset]


prob_tmp[i, j] =


# Store the computed probability matrix for the Bayesian network model under CV2 scheme:
prob_dict['cv2_' + k] = prob_tmp




# Printing probabilities
print(prob_dict['cv2_bn'])
print(prob_dict['cv2_pbn'])



 