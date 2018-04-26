# Building the feature matrix for the biomass:
index = ['loc', 'year', 'dap']
X_biomass = pd.get_dummies(df.loc[df.trait=='biomass', index])

# Adding the bin matrix to the feature matrix:
tmp = pd.get_dummies(df.id_gbs[df.trait=='biomass'])
X_biomass = np.hstack((np.dot(tmp, W_bin.loc[tmp.columns.tolist()]), X_biomass))

# Removing rows of the missing entries from the feature matrix:
X_biomass = X_biomass[np.invert(df.biomass[df.trait=='biomass'].isnull())]

# Creating a variable to receive the response without the missing values:
index = df.trait=='biomass'
y_biomass = df.biomass[index][np.invert(df.biomass[index].isnull())]

