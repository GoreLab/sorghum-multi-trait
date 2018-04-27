# Subsetting data into train and (dev set + test set) for biomass data:
X_biomass_trn, X_biomass_dev, y_biomass_trn, y_biomass_dev, index_cv['biomass_trn'], index_cv['biomass_dev'] = train_test_split(X_biomass, 
																														  y_biomass,
 		                                                																  df.drymass[index][np.invert(df.drymass[index].isnull())].index,
                                                        																  test_size=0.3,
                                                        																  random_state=1234)

# Subsetting (dev set + test set) into dev set and test set:
X_biomass_dev, X_biomass_tst, y_biomass_dev, y_biomass_tst, index_cv['biomass_dev'], index_cv['biomass_tst'] = train_test_split(X_biomass_dev,
	                                                            		  												  y_biomass_dev,
	                                                            		  												  index_cv['biomass_dev'],
                                                          				  												  test_size=0.50,
                                                          				  												  random_state=1234)

