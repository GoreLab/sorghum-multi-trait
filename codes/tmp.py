
#------------------------------------------------Modules-----------------------------------------------------#

## Loading libraries:
import matplotlib
# matplotlib.use('GTK') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns

import os

import tensorflow as tf
import pystan as ps
import subprocess
import dill
import time
import sys
import pickle
import re
import pprint

from scipy.stats import skew
from scipy.stats import moment
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score

# For adding flags to the code:
import argparse
parser = argparse.ArgumentParser()

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
# prefix_proj = "/home/jhonathan/Documents/sorghum-multi-trait/"
# prefix_proj = "/home/jhonathan/Documentos/sorghum-multi-trait/"
# prefix_proj = "/data1/aafgarci/jhonathan/sorghum-multi-trait/"
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documents/resul_mtrait-proj/"
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
# prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Setting directory:
os.chdir(prefix_proj + "codes")

# Loading external functions:
from external_functions import * 

# Loading inbred lines ID:
os.chdir(prefix_out + "data")

# Loading marker matrix:
M = pd.read_csv("gbs.csv", index_col=0)

# Changing index:
M.index = range(M.shape[0])

x = M.transpose()

#------------------------------------------Building bin function---------------------------------------------#



tmp = get_bin(x=x.iloc[:,:1000], n_bin=100, method='pca')

# Function to construct the bins:
def get_bin(x, n_bin, method):

n_bin = 50000


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


