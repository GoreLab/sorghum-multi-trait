

#----------------------------------------------Functions-----------------------------------------------------#

# Loading libraries:
import numpy
import os
import pandas

# Build the Cockerham's model (2, 1 and 0 centered marker scores):
def W_model(x):
  # Get allelic frequencies:
  p = (numpy.mean(x, axis=0)/2).values.reshape(1,(x.shape[1]))
  # Build Cokerham's model:
  W = x -  2*numpy.repeat(p, repeats= x.shape[0], axis=0)
  return(W)

# Construct the artificial bins:
def get_bin(x, n_bin, method):
  # Generate batches
  batches = numpy.array_split(numpy.arange(x.shape[1]), n_bin)
  # Initialize the binned matrix:
  W_bin = pandas.DataFrame(index=x.index, columns=map('bin_{}'.format, range(n_bin)))
  e_bin = []
  if method=='pca':
    for i in range(n_bin):
      # Compute SVD of the matrix bin:
      u,s,v = numpy.linalg.svd(x.iloc[:,batches[i]], full_matrices=False)
      # Compute the first principal component and adding to the binned matrix:
      W_bin['bin_' + str(i)] = numpy.dot(u[:,:1], numpy.diag(s[:1]))
      e_bin.append(s[0]/s.sum())
    return([W_bin, e_bin, batches])
  if method=='average':
    for i in range(n_bin):
      # Compute the mean across the columns and adding to the binned matrix:
      W_bin['bin_' + str(i)] = x.iloc[:,batches[i]].mean(axis=1)
      return([W_bin, batches])
