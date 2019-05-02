
# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Dictionary to receive cilabilities:
cil_dict = dict()

for k in model_set:
  for i in range(len(dap_group[:-1])):
    # Subset predictions and observations for cilability computation:
    y_pred_tmp = y_pred_fcv[k + '_fcv_height_trained!on!dap:' + dap_group[i]]
    y_obs_tmp = y_obs_fcv['fcv_height_for!trained!on:' + dap_group[i]].y_hat
    # Compute cilability across DAP measures:
    for j in range(len(dap_group)):
      # Conditional to compute cilability just forward in time:
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
        # Build cilability matrix for the Bayesian Network model:
        for sim in range(y_pred_tmp.shape[0]):
          # Top predicted lines:
          rank_pred = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1][0:n_selected]
          top_lines_pred = df.id_gbs[y_pred_tmp[subset].iloc[sim].iloc[rank_pred].index]
          # Compute the indicator of coincidence in the top 20%
          ind_tmp = top_lines_pred.isin(top_lines_obs)
          index_tmp = ind_tmp.iloc[np.where(ind_tmp)].index
          ind_vec.iloc[sim] = y_pred_tmp[subset].iloc[sim].index.isin(index_tmp)
        # Index to store cilabilties into dictionary:
        index = k + '_' + dap_group[i] + '_' + dap_group[j]
        # Compute cilability:
        cil_dict[index]=ind_vec.mean(axis=0)
        print('Model: {}, DAP_i: {}, DAP_j: {}'.format(k, dap_group[i], dap_group[j]))

