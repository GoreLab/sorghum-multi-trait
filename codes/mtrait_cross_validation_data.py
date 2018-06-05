
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
metrics = pd.DataFrame(index=(["height_" + str(i) for i in df.dap.unique()[1::]]).append('drymass'),
 					   columns=["cv"])
for i in df.trait.unique():
	for j in df.dap.unique()[1::]:
		if i=="height":
			metrics[i + '_' + str(j)] = pd.read_csv('metrics~' + i + '_' + str(j) + '-cv.csv', index_col=0)
		else:
			metrics[i] = pd.read_csv('metrics~' + i + '-cv.csv', index_col=0)

# Setting directory:
os.chdir(prefix_out + "data")

# Reading marker binned matrix:
W_bin = pd.read_csv("W_bin.csv", header = 0, index_col=0)

#--------------------------Splitting data into groups for 5th-fold cross-validation--------------------------#

# Number of folds:
n_fold = 5

# Creating five folds:
kf = KFold(n_splits=n_fold, shuffle=True, random_state=1234)

# Getting the splits:
index = kf.split(df['id_gbs'].drop_duplicates())

# Initializing lists to receive the random indexes:
trn_index = []
tst_index = []

# Getting the indexes:
for trn, tst in index:
	trn_index.append(df['id_gbs'].drop_duplicates().iloc[trn])
	tst_index.append(df['id_gbs'].drop_duplicates().iloc[tst])

# Creating dictionary with the data from the first cross-validation scheme:
y = dict()
X = dict()
for t in df.trait.unique():
	for i in range(n_fold):
		y[t + '_k' + i] = df.y_hat[df.id_gbs==trn_index[i]]


####3


tmp = pd.get_dummies(df.id_gbs[df.trait=='height'])

X['height'] = pd.concat([X['height'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)
