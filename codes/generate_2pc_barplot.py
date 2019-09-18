
#------------------------------------------------Modules-----------------------------------------------------#

# Import python modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# DATA_PATH = args.dpath
# REPO_PATH = args.rpath
# OUT_PATH = args.opath
DATA_PATH = '/workdir/jp2476/raw_data_sorghum-multi-trait'
REPO_PATH = '/workdir/jp2476/sorghum-multi-trait'
OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'


#----------------------------Generate histogram with percentage explained by the PCs-------------------------#

# Set directory:
os.chdir(DATA_PATH + "/raw_data")

# Load marker matrix:
M = pd.read_csv("gbs.csv")

# Load inbred lines ID:
line_names = pd.read_csv("genotype_names_corrected.csv")

# Load marker matrix:
loci_info = pd.read_csv("gbs_info.csv", index_col=0)

# Set the directory to store processed data:
os.chdir(OUT_PATH + "/processed_data")

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Intersection between IDs:
mask = ~df['id_gbs'].isnull()
line_names = np.intersect1d(np.unique(df['id_gbs'][mask].astype(str)), list(M))

# Ordering lines:
M = M.loc[:, line_names]

# Function to construct the bins:
def get_perc_pca(x, n_bin):
  # Generating batches
  batches = np.array_split(np.arange(x.shape[1]), n_bin)
  # Initializing the binned matrix:
  e_bin = pd.DataFrame(index=map('bin_{}'.format, range(n_bin)), columns=['pca1', 'pca2'])
  for i in range(n_bin):
    # Computing SVD of the matrix bin:
    u,s,v = np.linalg.svd(x.iloc[:,batches[i]], full_matrices=False)
    # Computing the first principal component and adding to the binned matrix:
    e_bin.loc['bin_' + str(i)]['pca1'] = (s[0]/s.sum())*100
    e_bin.loc['bin_' + str(i)]['pca2'] = (s[1]/s.sum())*100
  return e_bin

# Get percent explained by each bin:
perc_pca = get_perc_pca(M.transpose(), n_bin = 1000)

# Store index into a column:
perc_pca['Bin'] = perc_pca.index

# Melt data frame:
perc_pca = pd.melt(perc_pca, id_vars = 'Bin', var_name='pcs', value_name='perc')

# Change codification:
mask = perc_pca.pcs == 'pca1'
perc_pca.pcs[mask] = '1st PCA'
mask = perc_pca.pcs == 'pca2'
perc_pca.pcs[mask] = '2nd PCA'
perc_pca.columns = ['Bin', 'Principal Component', 'Percent explained (%)']

# Generate barplot with the percent explained by each bin:
sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.barplot(x="Bin", y='Percent explained (%)', hue='Principal Component', data=perc_pca)
g.despine(left=True)
# g.set_ylabels("survival probability")

# Set directory:
os.chdir(REPO_PATH + "/figures")

plt.savefig("perc_pca_expl.pdf", dpi=350)
plt.savefig("perc_pca_expl.png", dpi=350)
plt.clf()

