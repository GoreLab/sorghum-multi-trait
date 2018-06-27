


# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive the accuracy matrices:
prob_dict = dict()

# Compute correlation for the Baysian network and Pleiotropic Bayesian Network model under CV2 scheme:
for k in model_set:
	for i in range(len(dap_group[:-1])):
		# Subset predictions and observations for probability computation:
		y_pred_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]
		y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group[i]].y_hat
		# Computing probability across DAP measures:
		for j in range(len(dap_group)):
			# Conditional to compute probability just forward in time:
			if (j>i):
				# Subset indexes for subsetting the data:
				subset = df[df.dap==int(dap_group[j])].index
				# Get the number of selected individuals for 20% selection intensity:
				n_selected = int(y_obs_tmp[subset].size * 0.2)
				# Building the indexes for computing the coeficient of coincidence:
				rank_obs = np.argsort(y_obs_tmp[subset])[::-1][0:n_selected]
				# Vector for storing the indicators:
				ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
				# Build probability matrix for the Bayesian Network model:
				for sim in range(y_pred_tmp.shape[0]):
					# Get the indicator of which genotype is in the top 20% or not: 
					ind_vec.iloc[sim] = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1].isin(rank_obs)
				# Index to store probabilties into dictionary:
				index = k + '_' + dap_group[i] + '_' + dap_group[j]
				# Computing probability:
				prob_dict[index]=ind_vec.mean(axis=0)
				print('Model: {}, DAP_i: {}, DAP_j: {}'.format(k, dap_group[i], dap_group[j]))












																																																																																																																																					