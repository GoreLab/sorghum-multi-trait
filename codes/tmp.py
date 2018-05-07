# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['biomass_trn'].shape[0],
         p_x = X['biomass_trn'].shape[1],
         p_i = np.max(index_x),
         p_r = X['year'].shape[1],
         phi = np.max(y['biomass_trn'])*10,
         index_x = index_x,
         X = X['biomass_trn'],
         X_r = X['year'],
         y = y['biomass_trn'].flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['biomass_trn'] = model.sampling(data=df_stan, chains=1, iter=400)

