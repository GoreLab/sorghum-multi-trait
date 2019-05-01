
#------------------------------------------------Modules-----------------------------------------------------#

# Import python modules
import os
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

#-----------------------------------------Adding flags to the code-------------------------------------------#

# Get flags:
parser.add_argument("-dpath", "--dpath", dest = "dpath", help="The path of the folder with the raw data")
parser.add_argument("-rpath", "--rpath", dest = "rpath", help="The path of the repository")
parser.add_argument("-opath", "--opath", dest = "opath", help="The path of the folder to receive outputs (processed data)")

# Parse the paths:
args = parser.parse_args()

# Subset arguments:
DATA_PATH = args.dpath
REPO_PATH = args.rpath
OUT_PATH = args.opath
# DATA_PATH = '/home/jhonathan/Documents/raw_data_sorghum-multi-trait'
# REPO_PATH = '/home/jhonathan/Documents/sorghum-multi-trait'
# OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'


#--------------------------------------------Processing data-------------------------------------------------#

# Set directory:
os.chdir(REPO_PATH + "/clean_repository/codes")

# Import functions:
from functions import * 

# Set directory:
os.chdir(DATA_PATH + "/raw_data")

# Number of data frames:
n_df = 4

# Create an empty data frame:
df = []

# Load data:
df.append(pd.read_csv("Biomass_2016.csv"))
df.append(pd.read_csv("Biomass_SF2017.csv"))
df.append(pd.read_csv("heights_2016.csv"))
df.append(pd.read_csv("heights_SF2017.csv"))

# Get names of the data frame variables:
tmp = []
for i in range(n_df): tmp.append(list(df[i]))

# Rename the columns of the data frames:
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

# Add column mapping traits to the df:
df[0] = df[0].assign(trait=pd.Series(np.repeat('biomass', df[0].shape[0])).values)
df[1] = df[1].assign(trait=pd.Series(np.repeat('biomass', df[1].shape[0])).values)
df[2] = df[2].assign(trait=pd.Series(np.repeat('height', df[2].shape[0])).values)
df[3] = df[3].assign(trait=pd.Series(np.repeat('height', df[3].shape[0])).values)

# Add columns mapping years to the df:
df[0] = df[0].assign(year=pd.Series(np.repeat('16', df[0].shape[0])).values)
df[1] = df[1].assign(year=pd.Series(np.repeat('17', df[1].shape[0])).values)
df[2] = df[2].assign(year=pd.Series(np.repeat('16', df[2].shape[0])).values)

# The plant height data already have the year codification, just change the class for object:
df[3].year = df[3].year.astype(object)

# Sort data frames by plot ID:
df[0] = df[0].sort_values(['plot'])
df[1] = df[1].sort_values(['plot'])
df[2] = df[2].sort_values(['plot'])
df[3] = df[3].sort_values(['plot'])

# Index of for selecting columns of the df mapping the design related to biomass collected on 2016:
tmp = ['loc', 'set', 'block', 'range', 'row']

# Inclusion of the design variables to the combination height/2016
df[2] = pd.concat([df[2], df[0].loc[:,tmp]], sort=True, axis=1)

# Combine data frames into a unique data frame:
df = pd.concat(df, sort=True, axis=0)

# Change the index of the data frame:
df.index = np.arange(df.shape[0])

# Load inbred lines ID:
os.chdir(DATA_PATH + "/raw_data")
line_names = pd.read_csv("genotype_names_corrected.csv")

# Create an additional column in the data frame to receive the new identificator:
df = df.assign(id_gbs=pd.Series(np.repeat(np.nan, df.shape[0])).values)

# Get just the intersection of the same inbred line names:
line_names.index = line_names['Name2']
line_names = line_names.loc[np.intersect1d(line_names['Name2'], df['name2']),:]

# Unique name of the lines:
tmp=[]
tmp.append(line_names['Name2'].astype(str))
tmp.append(line_names['taxa'].astype(str))

# Add the inbred lines names to the data frame:
for i in range(len(tmp[0])):
	# Index mapping the presence of the correct inbred line name nto the data frame:
	index = df.name2.isin([tmp[0][i]])
	if np.sum(index) != 0:
		# Get the name of the line in the current iteration and finding it into the data frame:
		(df['id_gbs'])[np.squeeze(np.where(index))] = tmp[1][i]

# Load marker matrix:
M = pd.read_csv("gbs.csv")

# Load marker matrix:
loci_info = pd.read_csv("gbs_info.csv", index_col=0)

# Intersection between IDs:
line_names = np.intersect1d(np.unique(df['id_gbs'].astype(str)), list(M))

# Subset the inbred lines that we have phenotypes:
M = M.loc[:, line_names]

# Function to build the Cockerham's model:
W = W_model(x=M.transpose())

# Build the bin matrix:
tmp = get_bin(x=W, n_bin=1000, method='pca')
W_bin = tmp[0]

# Store the position of the bins into the genome:
bin_map = tmp[2]

# Add a new column for the data frame with the loci positions:
loci_info['bin'] = np.repeat(np.nan, loci_info.shape[0])

# Add the bins names for the bin column mapping its position:
for i in range(len(bin_map)):
	loci_info['bin'].iloc[bin_map[i]] = np.repeat(["bin_" + str(i)], bin_map[i].size)

# Subset only the inbred lines that were genotyped and phenotyped into the data frame:
tmp = np.where(df.id_gbs.isin(line_names))
df = df.loc[np.squeeze(tmp)]

# Name of the height covariates (columns that will be melt):
tmp = []
tmp.append(list(map('h{}'.format, range(1,8))))

# Name of all covariates except heights (variables that will be unaffected):
tmp.append(np.setdiff1d(list(df), tmp[0]))

# Change the shape of the data frame, and adding a new column mapping the days after planting (DAP)
df = pd.melt(df, id_vars=tmp[1], value_vars=tmp[0], var_name="dap", value_name="height")

# Replace categories by the values:
df['dap'] = df['dap'].replace(tmp[0], range(30, 120+15,15))

# Replace codification of locations:
df['loc'] = df['loc'].replace(['16EF', '16FF'], ['EF', 'FF'])

# Remove checks row from data not related to this experiment project:
df = df[df['block'] != 'CHK_STRP'][:]

# Change data type of the DAP:
df.dap = df.dap.astype(object)

# Average over the data structure except the factors evaluated in the multi trait project
df = df.groupby(['id_gbs', 'block', 'loc', 'year', 'trait', 'dap'], as_index=False).mean()

# Remove the DAP values from biomass, DAP values were taken only for plant height:
index = ((df.trait == 'biomass') & (df.dap == 120)) | (df.trait == 'height')
df = df[index] 

# Remove the DAP of 120 for biomass, it is not a right feature of biomass, actually it was collected in the end of the season:
index = (df.trait == 'biomass') & (df.dap == 120)
df.dap[index] = np.nan

# Remove the range column, it will not be used into the analysis:
df = df.drop('range', axis=1)

# Get just the features used in the paper:
df = df[['id_gbs', 'block', 'loc', 'year', 'trait', 'dap', 'drymass', 'height']]

# Changing traits codification:
df.trait[df.trait == 'biomass'] = 'DM'
df.trait[df.trait == 'height'] = 'PH'

# Changing the indexes of the data frame:
df.index = range(df.shape[0]) 

# Changing the data type of the 'year' feature:
df.year = df.year.astype(object)

## Writing into the disk the cleaned data:

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Writing the full data frame with phenotypic data and IDs:
df.to_csv("df.csv")

# Writing the bin info:
loci_info.to_csv("loci_info.csv")

# Writing the genomic binned matrix under Cockerham's model:						
W_bin.to_csv("W_bin.csv")

# Writing the full marker matrix:
M.to_csv("M.csv")