
#------------------------------------------------Modules-----------------------------------------------------#

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
import subprocess
import dill
import time
import sys 

from scipy.stats import skew
from scipy.stats import moment
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

# Prefix of the directory of the project is in:
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
prefix_proj = "/data1/aafgarci/jhonathan/mtrait-proj/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

#--------------------------------------------Processing data-------------------------------------------------#

# Setting directory:
os.chdir(prefix_out + "data")

# Number of data frames:
n_df = 4

# Creating an empty data frame:
df = []

# Loading data:
df.append(pd.read_csv("Biomass_2016.csv"))
df.append(pd.read_csv("Biomass_SF2017.csv"))
df.append(pd.read_csv("heights_2016.csv"))
df.append(pd.read_csv("heights_SF2017.csv"))

# Checking data types:
for i in range(n_df): df[i].dtypes

# Getting names of the data frame variables:
tmp = []
for i in range(n_df): tmp.append(list(df[i]))

# Renaming the columns of the data frames:
print(tmp[0])
df[0] = df[0].rename(index=str, columns={tmp[0][0]: 'loc',
		  							 	 tmp[0][1]: 'plot',
								 		 tmp[0][2]: 'name1',
								 		 tmp[0][3]: 'name2',
								 		 tmp[0][4]: 'set',
								 		 tmp[0][5]: 'range',
								 		 tmp[0][6]: 'row',
								 		 tmp[0][7]: 'block',
								 		 tmp[0][8]: 'moisture',
								 		 tmp[0][9]: 'drymass',
								 		 tmp[0][10]: 'starch',
								 		 tmp[0][11]: 'protein',
								 		 tmp[0][12]: 'adf',
								 		 tmp[0][13]: 'ndf'})

print(tmp[1])
df[1] = df[1].rename(index=str, columns={tmp[1][0]: 'plot',
		  							 	 tmp[1][1]: 'name1',
								 		 tmp[1][2]: 'name2',
								 		 tmp[1][3]: 'loc',
								 		 tmp[1][4]: 'row',
								 		 tmp[1][5]: 'range',
								 		 tmp[1][6]: 'set',
								 		 tmp[1][7]: 'block',
								 		 tmp[1][8]: 'date',
								 		 tmp[1][9]: 'moisture',
								 		 tmp[1][10]: 'drymass',
								 		 tmp[1][11]: 'starch',
								 		 tmp[1][12]: 'protein',
								 		 tmp[1][13]: 'adf',
								 		 tmp[1][14]: 'ndf'})


print(tmp[2])
df[2] = df[2].rename(index=str, columns={tmp[2][0]: 'plot',
		  							 	 tmp[2][1]: 'name1',
								 		 tmp[2][2]: 'name2',
								 		 tmp[2][3]: 'h1',
								 		 tmp[2][4]: 'h2',
								 		 tmp[2][5]: 'h3',
								 		 tmp[2][6]: 'h4',
								 		 tmp[2][7]: 'h5',
								 		 tmp[2][8]: 'h6',
								 		 tmp[2][9]: 'h7'})

print(tmp[3])
df[3] = df[3].rename(index=str, columns={tmp[3][0]: 'plot',
		  							 	 tmp[3][1]: 'name1',
								 		 tmp[3][2]: 'name2',
								 		 tmp[3][3]: 'taxa',
								 		 tmp[3][4]: 'year',
								 		 tmp[3][5]: 'loc',
								 		 tmp[3][6]: 'set',
								 		 tmp[3][7]: 'block',
								 		 tmp[3][8]: 'range',
								 		 tmp[3][9]: 'row',
								 		 tmp[3][10]: 'h1',
								 		 tmp[3][11]: 'h2',
								 		 tmp[3][12]: 'h3',
								 		 tmp[3][13]: 'h4',
								 		 tmp[3][14]: 'h5',
								 		 tmp[3][15]: 'h6',
								 		 tmp[3][16]: 'h7'})

# Adding column mapping traits to the df:
df[0] = df[0].assign(trait=pd.Series(np.repeat('biomass', df[0].shape[0])).values)
df[1] = df[1].assign(trait=pd.Series(np.repeat('biomass', df[1].shape[0])).values)
df[2] = df[2].assign(trait=pd.Series(np.repeat('height', df[2].shape[0])).values)
df[3] = df[3].assign(trait=pd.Series(np.repeat('height', df[3].shape[0])).values)

# Removing the year column of the unique data frame that have it:
df[3] = df[3].drop(['year'], axis=1)

# Adding columns mapping years to the df:
df[0] = df[0].assign(year=pd.Series(np.repeat('16', df[0].shape[0])).values)
df[1] = df[1].assign(year=pd.Series(np.repeat('17', df[1].shape[0])).values)
df[2] = df[2].assign(year=pd.Series(np.repeat('16', df[2].shape[0])).values)
df[3] = df[3].assign(year=pd.Series(np.repeat('17', df[3].shape[0])).values)

# Checking data types:
for i in range(n_df): df[i].dtypes

# Number of genotypes:
for i in range(n_df): len(np.unique(df[i]["name2"]))

# Number of plots:
for i in range(n_df): len(np.unique(df[i]["plot"]))

# Getting combination pairs:
tmp = list(itertools.combinations(range(0,n_df), 2))

# Printing the combinations:
print(tmp)

# Checking intersection of the plots across data sets:
for i,j in tmp: len(np.unique(np.intersect1d(df[i]["plot"], df[j]["plot"])))

# Lacking information on the height/2016:
print(np.setdiff1d(list(df[0]), list(df[2])))

# Sorting data frames by plot ID:
df[0] = df[0].sort_values(['plot'])
df[1] = df[1].sort_values(['plot'])
df[2] = df[2].sort_values(['plot'])
df[3] = df[3].sort_values(['plot'])

# Index of for selecting columns of the df mapping the design related to biomass collected on 2016:
tmp = ['loc', 'set', 'block', 'range', 'row']

# Test to see if the plots have the same ordering:
print(np.all(np.array(df[0]['plot']) == np.array(df[2]['plot']))) 

# Inclusion of the design variables to the combination height/2016
df[2] = pd.concat([df[2], df[0].loc[:,tmp]], axis=1)

# Combining data frames into a unique data frame:
df = pd.concat(df, axis=0)

# Changing the index of the data frame:
df.index = np.arange(df.shape[0])

# Checking if there is missing data into the design variables:
print(np.any(pd.isnull(df['loc'])))
print(np.any(pd.isnull(df['set'])))
print(np.any(pd.isnull(df['block'])))
print(np.any(pd.isnull(df['range'])))
print(np.any(pd.isnull(df['row'])))

# Loading inbred lines ID:
os.chdir(prefix_out + "data")
line_names = pd.read_csv("genotype_names_corrected.csv")

# Creating an additional column in the data frame to receive the new identificator:
df = df.assign(id_gbs=pd.Series(np.repeat(np.nan, df.shape[0])).values)

# Getting just the intersection of the same inbred line names:
line_names.index = line_names['Name2']
line_names = line_names.loc[np.intersect1d(line_names['Name2'], df['name2']),:]

# Unique name of the lines:
tmp=[]
tmp.append(line_names['Name2'].astype(str))
tmp.append(line_names['taxa'].astype(str))

# Adding the correct inbred lines names to the data frame:
for i in range(len(tmp[0])):
	# Index mapping the presence of the correct inbred line name nto the data frame:
	index = df.name2.isin([tmp[0][i]])
	if np.sum(index) != 0:
		# Getting the name of the line in the current iteration and finding it into the data frame:
		(df['id_gbs'])[np.squeeze(np.where(index))] = tmp[1][i]

# Loading marker matrix:
M = pd.read_csv("gbs.csv")

# Intersection between IDs:
line_names = np.intersect1d(np.unique(df['id_gbs'].astype(str)), list(M))

# Subsetting the inbred lines that we have phenotypes:
M = M.loc[:, line_names]

# Function to build the Cockerham's model:
W = W_model(x=M.transpose())

# Number of bins:
n_bin = 700

# Number of loci per bin:
n_loci_per_bin = int(W.shape[1]/n_bin)

# Building the bin matrix:
W_bin = get_bin(x=W, step_size=n_loci_per_bin)

# Transforming bin matrix into data frame and adding column index:
W_bin = pd.DataFrame(W_bin)

# Adding indexes:
W_bin.index = line_names.astype(str)
W_bin.columns = map('bin_{}'.format, range(W_bin.shape[1]))

# Removing M from memory by specifying an empty list:
M = []

# Subsetting only the inbred lines that were genotyped and phenotyped into the data frame:
tmp = np.where(df.id_gbs.isin(line_names))
df = df.loc[np.squeeze(tmp)]

# Name of the height covariates (columns that will be melt):
tmp = []
tmp.append(list(map('h{}'.format, range(1,8))))

# Name of all covariates except heights (variables that will be unaffected):
tmp.append(np.setdiff1d(list(df), tmp[0]))

# Changing the shape of the data frame, and adding a new column mapping the days after planting (DAP)
df = pd.melt(df, id_vars=tmp[1], value_vars=tmp[0], var_name="dap", value_name="height")

# Replacing categories by the values:
df['dap'] = df['dap'].replace(tmp[0], range(30, 120+15,15))

# Printing the number of entries per DAṔ:
df['dap'].value_counts()

# Printing the number of entries within each location:
df['loc'].value_counts()

# Replacing codification of locations:
df['loc'] = df['loc'].replace(['16EF', '16FF'], ['EF', 'FF'])

# Printing the number of entries within each block:
df['block'].value_counts()

# Removing checks row from data not related to this experiment project:
df = df[df['block'] != 'CHK_STRP'][:]

# Changing data type of the DAP:
df.dap = df.dap.astype(object)

# Averaging over the data structure except the factors evaluated in the multi trait project
df = df.groupby(['id_gbs', 'loc', 'year', 'trait', 'dap'], as_index=False).mean()

# Computing the mean of the numeric features:
tmp = df[df.trait == 'biomass'].mean()

# Index for subsetting rows:
index = (df.trait == 'biomass') & df.adf.isnull()

# Imputing adp:
df.adf[index] = np.repeat(tmp['adf'], np.sum(index))

# Index for subsetting rows:
index = (df.trait == 'biomass') & df.moisture.isnull()

# Imputing adp:
df.moisture[index] = np.repeat(tmp['moisture'], np.sum(index))

# Index for subsetting rows:
index = (df.trait == 'biomass') & df.ndf.isnull()

# Imputing adp:
df.ndf[index] = np.repeat(tmp['ndf'], np.sum(index))

# Index for subsetting rows:
index = (df.trait == 'biomass') & df.protein.isnull()

# Imputing adp:
df.protein[index] = np.repeat(tmp['protein'], np.sum(index))

# Index for subsetting rows:
index = (df.trait == 'biomass') & df.starch.isnull()

# Imputing adp:
df.starch[index] = np.repeat(tmp['starch'], np.sum(index))

## To do list:
# 1. Apply log transformation from nonnormal features, using the kaggle code
# 2. Design the cross-validation scheme
# 3. Ask the RNAseq data for Ravi






