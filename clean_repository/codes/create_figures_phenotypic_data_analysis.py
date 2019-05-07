
#------------------------------------------------Modules-----------------------------------------------------#

# Import python modules
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()

#-----------------------------------------Adding flags to the code-------------------------------------------#

# Get flags:
parser.add_argument("-rpath", "--rpath", dest = "rpath", help="The path of the repository")
parser.add_argument("-opath", "--opath", dest = "opath", help="The path of the folder to receive outputs")

# Parse the paths:
args = parser.parse_args()

# Subset arguments:
REPO_PATH = args.rpath
OUT_PATH = args.opath
# REPO_PATH = '/home/jhonathan/Documents/sorghum-multi-trait'
# OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'


#-----------------------------------------------Load data----------------------------------------------------#

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/codes")

# Import functions:
from functions import * 

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Change class of the dap:
df.dap = df.dap.fillna(0).astype(int)


#--------------------------------------------Generate Figures------------------------------------------------#

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

# Add a new column in the data frame for plotting:
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

# Density plot of the adjusted means from dry mass:
den_dm = sns.kdeplot(df.y_hat[df.trait=="drymass"], bw=1, shade=True, legend=False)
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

# # Set directory:
# os.chdir(OUT_PATH + "/heritabilities")

# # Read heritability values:
# h2_table = pd.read_csv('mtrait_first_step_analysis_heritability.txt', index_col=0)

# # Add new labels to the h2_table for plotting:
# h2_table['labels'] = labels
# h2_table

# # Add colors to be ploted:
# h2_table['colors'] = ['#006d2c', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506', '#fee391', '#fec44f']

# # Reorder the table:
# index = [0, 4, 5, 6, 7, 1, 2, 3]
# h2_table = h2_table.iloc[index]

# # Set directory:
# os.chdir(REPO_PATH + "/clean_repository/figures")

# # Plot heritabilities:
# bar_obj=plt.bar(h2_table['labels'].tolist(), h2_table['Estimate'].tolist(),
#  			    yerr = h2_table['SE'].tolist(),
#  			    align='center',
#  			    alpha=1,
#  			    color= h2_table['colors'].tolist()
#  			    )
# plt.tick_params(labelsize=7.6)
# plt.xticks(h2_table['labels'].tolist())
# plt.xticks(rotation=25)
# plt.xlabel('Traits')
# plt.ylabel('Broad-sense Heritability')
# plt.savefig("barplot_heritabilities.pdf", dpi=150)
# plt.savefig("barplot_heritabilities.png", dpi=150)
# plt.clf()


