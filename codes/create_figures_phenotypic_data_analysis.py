
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

# Turn off interactive mode:
plt.ioff()

#-----------------------------------------Adding flags to the code-------------------------------------------#

# Get flags:
parser.add_argument("-rpath", "--rpath", dest = "rpath", help="The path of the repository")
parser.add_argument("-opath", "--opath", dest = "opath", help="The path of the folder to receive outputs")

# Parse the paths:
args = parser.parse_args()

# Subset arguments:
REPO_PATH = args.rpath
OUT_PATH = args.opath
# REPO_PATH = '/workdir/jp2476/sorghum-multi-trait'
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'


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
labels = ["Dry biomass",
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
plt.savefig("heatplot_traits_adjusted_means.pdf", dpi=350)
plt.savefig("heatplot_traits_adjusted_means.png", dpi=350)
plt.clf()

# Density plot of the adjusted means from dry mass:
den_dm = sns.kdeplot(df.y_hat[df.trait=="drymass"], bw=1, shade=True, legend=False)
den_dm.set_ylabel('Density')    
den_dm.set_xlabel('Biomass (t/ha)')
den_dm.get_lines()[0].set_color('#006d2c')
x = den_dm.get_lines()[0].get_data()[0]
y = den_dm.get_lines()[0].get_data()[1]
plt.fill_between(x,y, color='#006d2c').set_alpha(.25)
plt.savefig("denplot_drymass_adjusted_means.pdf", dpi=350)
plt.savefig("denplot_drymass_adjusted_means.png", dpi=350)
plt.clf()

# Box plot of the adjusted means from height measures:
box_ph = sns.boxplot(x='dap', y='y_hat',
					 data=df[df.trait=="height"])
box_ph.set_ylabel('Height (cm)')    
box_ph.set_xlabel('Days after Planting')
colors = ['#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']

for i in range(1,len(colors)): 
	box_ph.artists[i].set_facecolor(colors[i])

plt.savefig("boxplot_height_adjusted_means.pdf", dpi=350)
plt.savefig("boxplot_height_adjusted_means.png", dpi=350)
plt.clf()

# Set directory:
os.chdir(OUT_PATH + "/heritabilities")

# Read heritability values:
h2_table = pd.read_csv('heritabilities.csv', index_col=0)

# Add new labels to the h2_table for plotting:
h2_table['trait'] = labels
h2_table

# Add colors to be ploted:
h2_table['colors'] = ['#006d2c', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506', '#fee391', '#fec44f']

# Reset index:
h2_table = h2_table.reset_index(drop=True)

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/figures")

# Plot heritabilities:
bar_obj=plt.bar(h2_table['trait'].tolist(), h2_table['h2'].tolist(),
 			    yerr = h2_table['se'].tolist(),
 			    align='center',
 			    alpha=1,
 			    color= h2_table['colors'].tolist()
 			    )
plt.tick_params(labelsize=7.6)
plt.xticks(h2_table['trait'].tolist())
plt.xticks(rotation=25)
plt.xlabel('Traits')
plt.ylabel('Broad-sense Heritability')
plt.savefig("barplot_heritabilities.pdf", dpi=350)
plt.savefig("barplot_heritabilities.png", dpi=350)
plt.clf()



