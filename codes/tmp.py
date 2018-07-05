





















# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Compute correlation for the Baysian network and Pleiotropic Bayesian Network model under CV2 scheme:
for k in model_set:
  for i in range(len(dap_group[:-1])):

k=model_set[0]
i=0


# Subset predictions and observations for probability computation:
expect_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]
y_obs_tmp = df[df.trait=='drymass'].y_hat

# Get the number of selected individuals for 20% selection intensity:
n_selected = int(y_obs_tmp.size * 0.2)

# Build the indexes for computing the coeficient of coincidence:
rank_obs = np.argsort(y_obs_tmp)[::-1][0:n_selected].index

# Selected lines:
top_lines_obs = df[df.trait=='drymass'].id_gbs[rank_obs]

# Vector for storing the indicators:
ind_vec = pd.DataFrame(index=expect_tmp.index, columns=expect_tmp.columns)

# Build probability matrix for the Bayesian Network model:
for sim in range(y_pred_tmp.shape[0]):
  # Get the ID of the predicted inbred lines for height:
  lines_pred = df.id_gbs[y_pred_tmp[subset].iloc[sim].index]
  # Get the indicator of which genotype is in the top 20% or not: 
  ind_vec.iloc[sim] = lines_pred.isin(top_lines_obs)

# Index to store probabilties into dictionary:
index = k + '_' + dap_group[i] + '_' + dap_group[j]

# Compute probability:
prob_dict[index]=ind_vec.mean(axis=0)
print('Model: {}, DAP_i: {}, DAP_j: {}'.format(k, dap_group[i], dap_group[j]))



