
# Index for subsetting height data:
index = df.trait=='height'

# Index to receive the position of the data frame:
index_cv = dict()

# Subsetting data into train and (dev set + test set) for height data:
X['height_trn'], X['height_dev'], y['height_trn'], y['height_dev'], index_cv['height_trn'], index_cv['height_dev'] = train_test_split(X['height'], 
																														  y['height'],
 		                                                																  df.height[index][np.invert(df.height[index].isnull())].index,
                                                        																  test_size=0.3,
                                                        																  random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X['height_dev'], X['height_tst'], y['height_dev'], y['height_tst'], index_cv['height_dev'], index_cv['height_tst'] = train_test_split(X['height_dev'],
	                                                            		  												  y['height_dev'],
	                                                            		  												  index_cv['height_dev'],
                                                          				  												  test_size=0.50,
                                                          				  												  random_state=1234)

# Index for subsetting height data:
index = df.trait=='biomass'

# Subsetting data into train and (dev set + test set) for biomass data:
X['biomass_trn'], X['biomass_dev'], y['biomass_trn'], y['biomass_dev'], index_cv['biomass_trn'], index_cv['biomass_dev'] = train_test_split(X['biomass'], 
																														        y['biomass'],
 		                                                																        df.drymass[index][np.invert(df.drymass[index].isnull())].index,
                                                        																        test_size=0.3,
                                                        																        random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X['biomass_dev'], X['biomass_tst'], y['biomass_dev'], y['biomass_tst'], index_cv['biomass_dev'], index_cv['biomass_tst'] = train_test_split(X['biomass_dev'],
	                                                            		  												        y['biomass_dev'],
	                                                            		  												        index_cv['biomass_dev'],
                                                          				  												        test_size=0.50,
                                                          				  												        random_state=1234)

# Reshaping responses:
y['height_trn'] = np.reshape(y['height_trn'], (y['height_trn'].shape[0], 1))
y['height_dev'] = np.reshape(y['height_dev'], (y['height_dev'].shape[0], 1))
y['height_tst'] = np.reshape(y['height_tst'], (y['height_tst'].shape[0], 1))
y['biomass_trn'] = np.reshape(y['biomass_trn'], (y['biomass_trn'].shape[0], 1))
y['biomass_dev'] = np.reshape(y['biomass_dev'], (y['biomass_dev'].shape[0], 1))
y['biomass_tst'] = np.reshape(y['biomass_tst'], (y['biomass_tst'].shape[0], 1))


# Checking shapes of the matrices related to height:
X['height_trn'].shape
y['height_trn'].shape
X['height_dev'].shape
y['height_dev'].shape
X['height_tst'].shape
y['height_tst'].shape

# Checking shapes of the matrices related to biomass:
X['biomass_trn'].shape
y['biomass_trn'].shape
X['biomass_dev'].shape
y['biomass_dev'].shape
X['biomass_tst'].shape
y['biomass_tst'].shape

#----------------------------Subdivision of the height data into mini-batches--------------------------------#

# Subsetting the full set of names of the inbred lines phenotyped for biomass:
index_mbatch = df.id_gbs[df.trait=='height'].drop_duplicates()

# Size of the mini-batch
mbatch_size = 4

# Splitting the list of names of the inbred lines into 4 sublists for indexing the mini-batches:
index_mbatch = list(split(index_mbatch, int(np.round(len(index_mbatch)/mbatch_size)) ))

## Indexing the mini-batches:

# Creating an empty list:
y['height_trn']_mb = []

index = df.id_gbs.loc[index_cv['height_trn']].isin(index_mbatch[0])


y['height_trn']_mb[0] = y['height_trn'][]

