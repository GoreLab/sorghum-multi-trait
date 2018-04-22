

#----------------------------------------------Functions-----------------------------------------------------#

# Loading libraries:
import numpy
import os
import pandas

## Function to construct bins:
def get_bin(x, step_size):
	# Renaming the array column names:
	x.columns = range(0,x.shape[1])
	# First step begin index:
	step_index = 0
	# Genome index:
	my_seq = numpy.arange(x.shape[1])
	# Infinity loop:
	var=1
	while var==1:
		# Index for averaging over the desired columns:
		index = numpy.intersect1d(my_seq, numpy.arange(start=step_index, stop=(step_index+step_size)))
		if my_seq[index].shape != (0,):
			# Averaging over columns:
			bin_tmp = numpy.mean(x.loc[:,my_seq[index]], axis=1).values.reshape([x.shape[0],1])
			# Stacking horizontally the bins:
			if step_index == 0:
				M_bin = bin_tmp
			else: 
				M_bin = numpy.hstack([M_bin, bin_tmp])
		# Updating the current step size:
		step_index = step_index + step_size
		if my_seq[index].shape == (0,):
		  break
	return(M_bin)


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
