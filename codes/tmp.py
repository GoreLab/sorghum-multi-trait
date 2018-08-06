



















# A list to receive the results:
results = [None] * n_alt

# Generating iterations:
setIter = range(1000)

# Generating epochs:
range_epoch = range(50)

# Small epsilon value for batch norm
epsilon = 1e-7

# Type of processor (CPU or GPU):
proc_type = "CPU"

# Number of hidden layers:
np.random.seed(seed)
n_layers_lst = sample_n_h_layers(min=1,          # Maximum number of hidden units
                                 max=20,         # Number of hidden layers
                                 n_guess=n_alt,  # Number of guesses
                                 same_str=2)     # False: Random guess; [Some architecture]: architecture to be replicated across guesses

# Batch norm (True or False):
np.random.seed(seed)
batch_mode_lst = np.random.choice([True, False], size=n_alt)

# Dropout (True or False):
np.random.seed(seed)
dropout_mode_lst = np.random.choice([True, False], size=n_alt) 

# Sampling the hidden units:
np.random.seed(seed)
h_units_lst =  sample_h_units(min=1,                    # Minimum number of hidden units
                              max=5,                    # Maximum number of hidden units
                              n_layers=n_layers_lst,    # Number of hidden layers (it should be a list)
                              n_guess=n_alt,            # Number of alternatives or guesses:
                              same_str=False)           # False: Random guess; [Some architecture]: architecture to be replicated across guesses

# Sampling the initial learning rate:
np.random.seed(seed)
starter_learning_rate_lst = sample_interval(min = 0.0001,        # Minimum of the quantitative interval
                                            max = 1,             # Maximum of the quantitative interval
                                            n_guess = n_alt,     # Number of guesses
                                            same_str = False)    # False: Random guess; [Value] insert a value to replicate

# Sampling Frobenius regularizer:
np.random.seed(seed)
lamb_lst = sample_interval(min = 0.0001,     # Minimum of the quantitative interval
                       max = 2,          # Maximum of the quantitative interval
                       n_guess = n_alt,  # Number of guesses
                       same_str = False) # False: Random guess; [Value] insert a value to replicate

# Sampling batch size:
np.random.seed(seed)
batch_size_lst = sample_batch(n = X['trn'].shape[1],        # Number of observations, or examples, or target
                              n_guess = n_alt,              # Number of guesses
                              same_str = X['trn'].shape[1]) # False: Random guess; [Value] insert a value to replicate

# Sampling dropout keep prob hyperparameter:
np.random.seed(seed)
keep_prob_lst = sample_interval(min = 0.0001,      # Minimum of the quantitative interval
                                max = 1,           # Maximum of the quantitative interval
                                n_guess = n_alt,   # Number of guesses
                                same_str = False)  # False: Random guess; [Value] insert a value to replicate



###############


m=6
n_alt=40


seed = int(str(m) + str(n_alt))


np.random.seed(seed)


# A list to receive the results:
results = [None] * n_alt

# Generating iterations:
setIter = range(1000)

# Generating epochs:
range_epoch = range(50)

# Small epsilon value for batch norm
epsilon = 1e-7

# Type of processor (CPU or GPU):
proc_type = "CPU"

# Number of hidden layers:
# np.random.seed(seed)
n_layers_lst = sample_n_h_layers(min=1,          # Maximum number of hidden units
                                 max=20,         # Number of hidden layers
                                 n_guess=n_alt,  # Number of guesses
                                 same_str=2)     # False: Random guess; [Some architecture]: architecture to be replicated across guesses

# Batch norm (True or False):
# np.random.seed(seed)
batch_mode_lst = np.random.choice([True, False], size=n_alt)

# Dropout (True or False):
# np.random.seed(seed)
dropout_mode_lst = np.random.choice([True, False], size=n_alt) 

# Sampling the hidden units:
# np.random.seed(seed)
h_units_lst =  sample_h_units(min=1,                    # Minimum number of hidden units
                              max=5,                    # Maximum number of hidden units
                              n_layers=n_layers_lst,    # Number of hidden layers (it should be a list)
                              n_guess=n_alt,            # Number of alternatives or guesses:
                              same_str=False)           # False: Random guess; [Some architecture]: architecture to be replicated across guesses

# Sampling the initial learning rate:
# np.random.seed(seed)
starter_learning_rate_lst = sample_interval(min = 0.0001,        # Minimum of the quantitative interval
                                            max = 1,             # Maximum of the quantitative interval
                                            n_guess = n_alt,     # Number of guesses
                                            same_str = False)    # False: Random guess; [Value] insert a value to replicate

# Sampling Frobenius regularizer:
# np.random.seed(seed)
lamb_lst = sample_interval(min = 0.0001, # Minimum of the quantitative interval
                       max = 2,          # Maximum of the quantitative interval
                       n_guess = n_alt,  # Number of guesses
                       same_str = False) # False: Random guess; [Value] insert a value to replicate

# Sampling batch size:
# np.random.seed(seed)
batch_size_lst = sample_batch(n = X['trn'].shape[1],        # Number of observations, or examples, or target
                              n_guess = n_alt,              # Number of guesses
                              same_str = X['trn'].shape[1]) # False: Random guess; [Value] insert a value to replicate

# Sampling dropout keep prob hyperparameter:
# np.random.seed(seed)
keep_prob_lst = sample_interval(min = 0.0001,      # Minimum of the quantitative interval
                                max = 1,           # Maximum of the quantitative interval
                                n_guess = n_alt,   # Number of guesses
                                same_str = False)  # False: Random guess; [Value] insert a value to replicate



print(batch_mode_lst)
print(dropout_mode_lst)

print(n_layers_lst)
print(h_units_lst)

print(starter_learning_rate_lst)
print(lamb_lst)
print(batch_size_lst)
print(keep_prob_lst)


print(batch_mode_lst[6])
print(dropout_mode_lst[6])

print(n_layers_lst[6])
print(h_units_lst[6])

print(starter_learning_rate_lst[6])
print(lamb_lst[6])
print(batch_size_lst[6])
print(keep_prob_lst[6])


>>> print(batch_mode_lst[6])
False
>>> print(dropout_mode_lst[6])
False
>>> 
>>> print(n_layers_lst[6])
2
>>> print(h_units_lst[6])
[4 4 1]
>>> 
>>> print(starter_learning_rate_lst[6])
0.1809194251820892
>>> print(lamb_lst[6])
0.0005156945746242638
>>> print(batch_size_lst[6])
26870
>>> print(keep_prob_lst[6])
0.0006587624926100591
