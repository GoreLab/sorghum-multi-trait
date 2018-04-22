# Adding column mapping traits to the df:
df[0] = df[0].assign(trait=pd.Series(np.repeat('biomass', df[0].shape[0])).values)
df[1] = df[1].assign(trait=pd.Series(np.repeat('height', df[1].shape[0])).values)
df[2] = df[2].assign(trait=pd.Series(np.repeat('biomass', df[2].shape[0])).values)
df[3] = df[3].assign(trait=pd.Series(np.repeat('height', df[3].shape[0])).values)

# Removing the year column of the unique data frame that have it:
df[3].drop(['year'], axis=1)

# Adding columns mapping traits to the df:
df[0] = df[0].assign(trait=pd.Series(np.repeat('biomass', df[0].shape[0])).values)
df[1] = df[1].assign(trait=pd.Series(np.repeat('height', df[1].shape[0])).values)
df[2] = df[2].assign(trait=pd.Series(np.repeat('biomass', df[2].shape[0])).values)
df[3] = df[3].assign(trait=pd.Series(np.repeat('height', df[3].shape[0])).values)