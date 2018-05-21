----------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
# matplotlib.use('GTK') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns

import os

import tensorflow as tf
import pystan as ps
import subprocess
import dill
import time
import sys
import re 

from scipy.stats import skew
from scipy.stats import moment
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

# For adding flags to the code:
import argparse
parser = argparse.ArgumentParser()

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documents/sorghum-multi-trait/"
# prefix_proj = "/home/jhonathan/Documentos/sorghum-multi-trait/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documents/resul_mtrait-proj/"
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

#---------------------------------------Computing correlations-----------------------------------------------#


# Setting the directory:
os.chdir(prefix_out + "data")

# Loading the data frame with phenotypic data and id's:
df = pd.read_csv("df.csv", header = 0, index_col=0)

# Changing the class of the year column:
df.year = df.year.astype(object)

# Traits list to correlate:
traits = df.columns.tolist()[5:]

# DAP to correlate:
daps = df.dap.drop_duplicates()[1:]

# Computing means of genotypes across all data structure:
tmp = pd.DataFrame(index=df.id_gbs.drop_duplicates())
for k in traits:
	if k == 'height':
		for i in daps:
			# Averaging over data structure except the trait:
			tmp['Height DAP ' + str(int(i))] = np.array(df[df.dap==i][['id_gbs','height']].groupby(['id_gbs']).mean())
	else:
		# Averaging over data structure except the trait:
		tmp[k.title()] = np.array(df[['id_gbs', k]].groupby(['id_gbs']).mean())

# Computing correlations:
corr = tmp.corr()


# Setting directory:
os.chdir(prefix_proj + "plots")

# Plotting correlations:
heat = sns.heatmap(round(corr,2), annot=True)
plt.xticks(rotation=25)

# Saving Figure:
plt.savefig("correlation_traits_heatmap_plot.png")

# Cleaning memory:
plt.clf()
