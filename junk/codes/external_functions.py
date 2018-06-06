

#----------------------------------------------Functions-----------------------------------------------------#

# Loading libraries:
import numpy
import os
import pandas

## Yield successive n-sized chunks from x
def split(x, n):
    for i in range(0, len(x), n):
        yield x[i:i + n]

# Function to construct the bins:
def get_bin(x, n_bin, method):
	# Generating batches
	batches = numpy.array_split(numpy.arange(x.shape[1]), n_bin)
	# Initializing the binned matrix:
	W_bin = pandas.DataFrame(index=x.index, columns=map('bin_{}'.format, range(n_bin)))
	e_bin = []
	if method=='pca':
		for i in range(n_bin):
			# Computing SVD of the matrix bin:
			u,s,v = numpy.linalg.svd(x.iloc[:,batches[i]], full_matrices=False)
			# Computing the first principal component and adding to the binned matrix:
			W_bin['bin_' + str(i)] = numpy.dot(u[:,:1], numpy.diag(s[:1]))
			e_bin.append(s[0]/s.sum())
		return([W_bin, e_bin])
	if method=='average':
		for i in range(n_bin):
			# Computing the mean across the columns and adding to the binned matrix:
			W_bin['bin_' + str(i)] = x.iloc[:,batches[i]].mean(axis=1)
			return(W_bin)

## Function to build the Cockerham's model:
def W_model(x):
	# Getting allelic frequencies:
	p = (numpy.mean(x, axis=0)/2).values.reshape(1,(x.shape[1]))
	# Building Cokerham's model:
	W = x -  2*numpy.repeat(p, repeats= x.shape[0], axis=0)
	return(W)

## Function for sampling hidden units:
def sample_h_units(min, max, n_layers, n_guess, same_str):
	x = []
	if same_str==False:
		# Generating random numbers:
		for j in range(n_guess):
				   	x.append(numpy.append(numpy.random.choice(range(min,max+1), size=n_layers[j]), 1.).astype(int))
		return(x)
	if same_str != False:
		return(numpy.tile(same_str, [n_guess, 1]))


## Function to guess the initial learning rate values:
def sample_interval(min, max, n_guess, same_str):
		if same_str==False:
			min, max = numpy.log10(.00001), numpy.log10(1)                     # Uniform distribution parameters
			lamb_lst = 10**(min + (max-min)*numpy.random.rand(n_guess))     # Regularization parameter
			return(lamb_lst)
		if same_str!=False:
			return(numpy.repeat(same_str, n_guess))


## Function to choose the batch size:
def sample_batch(n, n_guess, same_str):
		if same_str==False:
				return(numpy.random.choice([n, 32, 64, 256, 512], size=n_guess))
		if same_str!=False:
				return(numpy.repeat(same_str, n_guess))


## Function to choose the batch size:
def sample_n_h_layers(min, max, n_guess, same_str):
		if same_str==False:
				return(numpy.random.choice(range(min,max+1), size=n_guess))
		if same_str!=False:
				return(numpy.repeat(same_str, n_guess))

# Function to compute RMSE metrics:
def rmse(y_pred, y):
    return numpy.sqrt(((y_pred - y) ** 2).mean())


# Function to get some index set of n minimum values in a 2D array:
def get_index(array, n):
    # Getting minimum values from 1d array:
    tmp = array.argsort(axis=None)[:n]
    # Getting minimum values:
    tmp = numpy.ndarray.flatten(array)[tmp]
    # Getting indexes:
    index = numpy.where(pandas.DataFrame(array).isin(tmp))
    return numpy.squeeze(index)
