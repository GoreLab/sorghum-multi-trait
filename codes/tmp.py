












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
    return([W_bin, e_bin, batches])
  if method=='average':
    for i in range(n_bin):
      # Computing the mean across the columns and adding to the binned matrix:
      W_bin['bin_' + str(i)] = x.iloc[:,batches[i]].mean(axis=1)
      return([W_bin, batches])


