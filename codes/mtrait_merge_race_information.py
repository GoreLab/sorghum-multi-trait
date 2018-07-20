
#--------------------------------------------Load libraries---------------------------------------------------#

# Import libraries:
import os
import pandas as pd
import numpy as np

#-----------------------------------------------Read data-----------------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs was saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Set directory:
os.chdir(prefix_out + "outputs/first_step_analysis")

# Read adjusted means:
df = pd.read_csv("adjusted_means.csv", index_col=0)

# Set directory:
os.chdir(prefix_proj + 'data_small_files')

# Read races codifications:
race_grin = pd.read_csv('race_grin.csv', header = 0)
race_cod = pd.read_csv('races_codification.csv', header = 0)
race_terra = pd.read_csv('ultimate_sorghum.csv', header = 0)


#---------------------------------------------Merge databases-------------------------------------------------#

# Adding a new column mapping the full inbreds id:
race_grin['id_gbs'] = race_grin['accession_prefix'].map(str) + race_grin['accession_number'].map(str)

# Subset just desired columns:
race_grin = race_grin[['id_gbs', 'observation_value']]
race_grin.columns = ['id_gbs', 'race_code']

# Adding a new column to the relational table:
race_grin['race'] = np.nan

# Adding the name of the races to the relational table:
for i in range(race_cod.shape[0]):
  # Mask for subset:
  mask = race_grin['race_code'] == race_cod['code'][i]
  # Subset the race based on code:
  race_grin['race'][mask] = race_cod['race'][i]
  print('Iter {} of {}'.format(i,race_cod.shape[0]))

# Filter commom acessions:
mask = race_grin['id_gbs'].isin(df['id_gbs'].drop_duplicates())
race_grin = race_grin[mask]

# Drop race code column and drop duplicates:
race_grin = race_grin.drop(['race_code'], axis=1).drop_duplicates()

# Filter races that does not overlap between databases:
mask = ~(race_terra[~race_terra['Race'].isnull()]['ID'].isin(race_grin['id_gbs']))
race_non_common = race_terra[~race_terra['Race'].isnull()][mask]
race_non_common = race_non_common[['ID', 'Race']]
race_non_common.columns = ['id_gbs', 'race']

# Adding non common races to the grin database and create a new one:
mask = race_non_common['id_gbs'].isin(df['id_gbs'].drop_duplicates())
race_new = pd.concat([race_grin, race_non_common[mask]], axis=0)

# Write into disk the new race file:
race_new.to_csv("race_new.csv")
