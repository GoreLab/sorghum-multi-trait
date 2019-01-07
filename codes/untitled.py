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


