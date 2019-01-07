
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
         cmap='YlOrBr',
         vmin=0.09,
         vmax=1,
         annot=True,
         annot_kws={"size": 12},
         xticklabels=labels ,
         yticklabels=labels)
heat.set_ylabel('')    
heat.set_xlabel('')
heat.tick_params(labelsize=7.6)
plt.xticks(rotation=25)
plt.yticks(rotation=45)
plt.savefig("heatplot_traits_adjusted_means.pdf", dpi=150)
plt.savefig("heatplot_traits_adjusted_means.png", dpi=150)
plt.clf()

# Density plot of the adjusted means from dry mass (2.241699: from US t/acre to t/ha):
den_dm = sns.kdeplot(df.y_hat[df.trait=="drymass"]*2.241699, bw=1, shade=True, legend=False)
den_dm.set_ylabel('Density')    
den_dm.set_xlabel('Biomass (t/ha)')
den_dm.get_lines()[0].set_color('#006d2c')
x = den_dm.get_lines()[0].get_data()[0]
y = den_dm.get_lines()[0].get_data()[1]
plt.fill_between(x,y, color='#006d2c').set_alpha(.25)
plt.savefig("denplot_drymass_adjusted_means.pdf", dpi=150)
plt.savefig("denplot_drymass_adjusted_means.png", dpi=150)
plt.clf()

# Box plot of the adjusted means from height measures:
box_ph = sns.boxplot(x='dap', y='y_hat',
					 data=df[df.trait=="height"])
box_ph.set_ylabel('Height (cm)')    
box_ph.set_xlabel('Days after Planting')
colors = ['#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']

for i in range(1,len(colors)): 
	box_ph.artists[i].set_facecolor(colors[i])

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
metrics

# Plot CVs:
bar_cv = sns.barplot(x='labels', y='cv', data=metrics)
bar_cv.set(xlabel='Traits', ylabel='Coefficient of variation (%)')
plt.xticks(rotation=25) 
bar_cv.tick_params(labelsize=6)
plt.savefig("barplot_coefficient_of_variation.pdf", dpi=150)
plt.savefig("barplot_coefficient_of_variation.png", dpi=150)
plt.clf()

# Read heritability values:
h2_table = pd.read_csv('mtrait_first_step_analysis_heritability.txt', index_col=0)

# Add new labels to the h2_table for plotting:
h2_table['labels'] = labels
h2_table

# Add colors to be ploted:
h2_table['colors'] = ['#006d2c', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506', '#fee391', '#fec44f']

# Plot heritabilities:
# bar_cv = sns.barplot(x='labels', y='h2', data=h2_table)
# bar_cv.set(xlabel='Traits', ylabel='Broad-sense Heritability')
# plt.xticks(rotation=25)
# bar_cv.tick_params(labelsize=6)
# plt.savefig("barplot_heritabilities.pdf", dpi=150)
# plt.savefig("barplot_heritabilities.png", dpi=150)
# plt.clf()

# Reordering the table:
index = [0, 4, 5, 6, 7, 1, 2, 3]
h2_table = h2_table.iloc[index]

# Plot heritabilities:
bar_obj=plt.bar(h2_table['labels'].tolist(), h2_table['Estimate'].tolist(),
 			    yerr = h2_table['SE'].tolist(),
 			    align='center',
 			    alpha=1,
 			    color= h2_table['colors'].tolist()
 			    )
plt.tick_params(labelsize=7.6)
plt.xticks(h2_table['labels'].tolist())
plt.xticks(rotation=25)
plt.xlabel('Traits')
plt.ylabel('Broad-sense Heritability')
plt.savefig("barplot_heritabilities.pdf", dpi=150)
plt.savefig("barplot_heritabilities.png", dpi=150)
plt.clf()


