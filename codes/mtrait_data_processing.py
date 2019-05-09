
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

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

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
for i, j in tmp: len(np.unique(np.intersect1d(df[i]["plot"], df[j]["plot"])))

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

# Loading marker matrix:
loci_info = pd.read_csv("gbs_info.csv", index_col=0)

# Intersection between IDs:
line_names = np.intersect1d(np.unique(df['id_gbs'].astype(str)), list(M))

# Subsetting the inbred lines that we have phenotypes:
M = M.loc[:, line_names]

# Function to build the Cockerham's model:
W = W_model(x=M.transpose())

# Building the bin matrix:
tmp = get_bin(x=W, n_bin=1000, method='pca')
W_bin = tmp[0]
rownames(W_bin) = colnames(M)
 
# Store the variação explained by the pca:
w_e_bin = tmp[1]

# Store the position of the bins into the genome:
bin_map = tmp[2]

# Adding a new column for the data frame with the loci positions:
loci_info['bin'] = np.repeat(np.nan, loci_info.shape[0])

# Adding the bins names for the bin column mapping its position:
for i in range(len(bin_map)):
	loci_info['bin'].iloc[bin_map[i]] = np.repeat(["bin_" + str(i)], bin_map[i].size)

# Removing M from memory:
# M = None

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

# Reading RNAseq transcriptomic data:
rnaseq = pd.read_table("df_STAR_HTSeq_counts_sbDiverse_DESeq2_vstNormalized.txt", index_col=0)

# Getting the inbred lines names that we have RNAseq data:
tmp = []
tmp.append(np.unique(rnaseq.columns.str.split('_').str.get(0)))

# Matrix of the transcriptomic data:
T = np.array((rnaseq.filter(like=tmp[0][0], axis=1).mean(axis=1)))
T = T.reshape([T.shape[0], 1])
tmp.append(T.shape)

# Averaging over tissues types:
for i in range(1,len(tmp[0])): T = np.concatenate([T, rnaseq.filter(like=tmp[0][i], axis=1).mean(axis=1).values.reshape(tmp[1])], axis=1)

# Transforming to pandas type:
T = pd.DataFrame(T.transpose(), index=tmp[0], columns=rnaseq.index)

# Indexing just the IDs where there is both genotypic, phenotypic and transcriptomic data:
tmp = df[df.name1.isin(T.index)][['name1', 'id_gbs']].drop_duplicates()

# Reordering the transcriptomic RNAseq data, and indexing just phenotyped and genotyped individuals:
T = T.loc[tmp.name1]

# Replacing by the name1 index to the id_gbs index:
T.index = tmp.id_gbs

# Building the bin matrix:
tmp = get_bin(x=T, n_bin=1000, method='pca')
T_bin = tmp[0]

# Storing the variação explained by the pca:
t_e_bin = tmp[1]

# Removing T from memory:
T = None

# Averaging over the data structure except the factors evaluated in the multi trait project
df = df.groupby(['id_gbs', 'block', 'loc', 'year', 'trait', 'dap'], as_index=False).mean()

# Removing the DAP values from biomass, DAP values were taken only for plant height:
index = ((df.trait == 'biomass') & (df.dap == 120)) | (df.trait == 'height')
df = df[index] 

# Removing the DAP of 120 for biomass, it is not a right feature of biomass, actually it was collected in the end of the season:
index = (df.trait == 'biomass') & (df.dap == 120)
df.dap[index] = np.nan

# # Computing the mean of the numeric features:
# tmp = df[df.trait == 'biomass'].mean()

# # Imputing adp:
# index = (df.trait == 'biomass') & df.adf.isnull()
# df.adf[index] = np.repeat(tmp['adf'], np.sum(index))

# # Imputing moisture:
# index = (df.trait == 'biomass') & df.moisture.isnull()
# df.moisture[index] = np.repeat(tmp['moisture'], np.sum(index))

# # Imputing ndf:
# index = (df.trait == 'biomass') & df.ndf.isnull()
# df.ndf[index] = np.repeat(tmp['ndf'], np.sum(index))

# # Imputing protein:
# index = (df.trait == 'biomass') & df.protein.isnull()
# df.protein[index] = np.repeat(tmp['protein'], np.sum(index))

# # Imputing starch:
# index = (df.trait == 'biomass') & df.starch.isnull()
# df.starch[index] = np.repeat(tmp['starch'], np.sum(index))

# Removing the range column, it will not be used into the analysis:
df = df.drop('range', axis=1)

# # Traits list:
# tmp = ['adf', 'drymass', 'moisture', 'ndf', 'protein', 'starch', 'height']

# # Reordering the columns of the data frame:
# tmp = df.columns.tolist()
# tmp = tmp[0:5] + [tmp[5]] + tmp[7:11] + [tmp[6]] + [tmp[-1]]
# df = df[tmp]

# Getting just the features used in the paper:
df = df[['id_gbs', 'block', 'loc', 'year', 'trait', 'dap', 'drymass', 'height']]

# Changing traits codification:
df.trait[df.trait == 'biomass'] = 'DM'
df.trait[df.trait == 'height'] = 'PH'

# Changing the indexes of the data frame:
df.index = range(df.shape[0]) 

# Changing the data type of the 'year' feature:
df.year = df.year.astype(object)

## Writing into the disk the cleaned data:

# Writing the full data frame with phenotypic data and IDs:
df.to_csv("df.csv")

# Writing the bin info:
loci_info.to_csv("loci_info.csv")

# Writing the genomic binned matrix under Cockerham's model:						
W_bin.to_csv("W_bin.csv")

# Writing proportion explained by the bin:
pd.DataFrame(w_e_bin, index=W_bin.columns).to_csv("w_e_bin.csv")

# Writing the transcriptomic binned matrix:
T_bin.to_csv("T_bin.csv")

# Writing proportion explained by the bin:
pd.DataFrame(t_e_bin, index=T_bin.columns).to_csv("t_e_bin.csv")

# Writing the full marker matrix:
M.to_csv("M.csv")
