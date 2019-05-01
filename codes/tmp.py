
#----Compute coincidence index for dry biomass indirect selection using height adjusted means time series-----#

# Store into a list different DAP values:
dap_group = ['30', '45', '60', '75', '90', '105', '120']

# List of models to use:
model_set = ['bn', 'pbn']

# Compute coincidence index for the Bayesian network and Pleiotropic Bayesian Network model:
for k in model_set:
  for i in range(len(dap_group[:-1])):
    # Subset expectations and observations for coincidence index computation:
    expect_tmp = expect_fcv[k + '_fcv_height_trained!on!dap:' + dap_group[i]]
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
  expect_tmp = expect_fcv['dbn_fcv_height_trained!on!dap:' + dap_group[i]]
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

# Change data type:
ci_post['post'] = ci_post['post'].astype(float)

# Change labels:
ci_post.columns = ['Days after planting', 'Model', 'Coincidence index posterior values']

# Set directory:
os.chdir(prefix_proj + 'plots/cv/ciplot')

# Plot coincidence indexes:
ax = sns.violinplot(x='Days after planting',
                    y='Coincidence index posterior values',
                    data=ci_post,
                    hue='Model')
plt.ylim(0.12, 0.42)
plt.savefig("ci_plot.pdf", dpi=150)
plt.savefig("ci_plot.png", dpi=150)
plt.clf()

