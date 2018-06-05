
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns

import os

from sklearn.model_selection import KFold

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 


#-----------------------------------Loading Adjusted Means and plotting--------------------------------------#

# Setting directory:
os.chdir(prefix_out + "outputs/first_step_analysis")

# Readomg adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Changing class of the dap:
df.dap = df.dap.fillna(0).astype(int)

# Adding a new column in the data frame for plotting:
tmp = df.pivot(index='id_gbs', columns='dap', values='y_hat')

# Labels for plotting the heatmap:
labels = ["Biomass",
  		  "Height DAP 30",
		  "Height DAP 45",
		  "Height DAP 60",
		  "Height DAP 75",
		  "Height DAP 90",
		  "Height DAP 105",
		  "Height DAP 120"]

# Heat map of the adjusted means across traits:
heat = sns.heatmap(tmp.corr(),
 				   linewidths=0.25,
 	 			   annot=True,
 				   annot_kws={"size": 10},
 				   xticklabels=labels,
 				   yticklabels=labels)
heat.set_ylabel('')    
heat.set_xlabel('')
plt.xticks(rotation=25)
heat.tick_params(labelsize=6)
plt.savefig("heatplot_traits_adjusted_means.pdf", dpi=150)
plt.savefig("heatplot_traits_adjusted_means.png", dpi=150)
plt.clf()

# Density plot of the adjusted means from dry mass:
den_dm = sns.kdeplot(df.y_hat[df.trait=="drymass"], bw=0.5, shade=True, legend=False)
den_dm.set_ylabel('Density')    
den_dm.set_xlabel('Dry mass (units)')
plt.savefig("denplot_drymass_adjusted_means.pdf", dpi=150)
plt.savefig("denplot_drymass_adjusted_means.png", dpi=150)
plt.clf()

# Box plot of the adjusted means from height measures:
box_ph = sns.boxplot(x='dap', y='y_hat',
					 data=df[df.trait=="height"])
box_ph.set_ylabel('Height (cm)')    
box_ph.set_xlabel('Days after Planting')
plt.savefig("boxplot_height_adjusted_means.pdf", dpi=150)
plt.savefig("boxplot_height_adjusted_means.png", dpi=150)
plt.clf()

# Loading coeficient of variation values:
metrics = np.empty([df.dap.unique().size])
metrics[:] = np.nan
counter = 0
for j in df.dap.unique():
	for i in df.trait.unique():
		if (i=="height") & (j!=0):
			metrics[counter] = pd.read_csv('metrics~' + i + '_' + str(j) + '-cv.csv', index_col=0).get_values()*100
		if (i=="drymass") & (j==0):
			metrics[counter] = pd.read_csv('metrics~' + i + '-cv.csv', index_col=0).get_values()*100
	counter = counter + 1

# Transform into pandas data frame:
metrics = pd.DataFrame(metrics, columns=["cv"]).assign(labels=labels)

# Plotting CVs:
bar_cv = sns.barplot(x='labels', y='cv', data=metrics)
bar_cv.set(xlabel='Traits', ylabel='Coefficient of variation (%)')
plt.xticks(rotation=25)
plt.savefig("barplot_coefficient_of_variation.pdf", dpi=150)
plt.savefig("barplot_coefficient_of_variation.png", dpi=150)
plt.clf()
