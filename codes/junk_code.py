
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#---------------------------------------Heritability based on PEV--------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Function to compute heritability:
get_h2_pev = function(fit) {

  # Get the PEV:
  pred = predict(fit, classify = 'id_gbs', sed=TRUE)
  pev <- unname(pred$predictions$avsed[2]^2)
  
  # Compute heritability:
  h2 <- 1-(pev/(2*summary(fit, all=T)$varcomp['id_gbs!id_gbs.var',2]))

  return(h2)

}


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#-------------------------------------Logistic Growth function test------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#


## Logistic growth function:
def logistic_growth(a, c, r, t):
    return(c / (1 + a * np.exp(-r * t)))


c = np.repeat(100, 3)
a = np.repeat(10, 3)
r = np.repeat(4, 3)

out = np.empty((3, 7))
out[:] = np.nan

time = np.linspace(30,120,7)/30.0

counter=0
for t in time:
  out[:,counter] = logistic_growth(a,c,r,t)
  counter = counter + 1

plt.plot(out[1,])


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#-------------------------------Asreml multivariate linear mixed model analysis------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)
library(rrBLUP)

# Function for filtering for maf:
filter_maf= function(X, cutoff) {

	# Compute allelic frequency for codification 2 (AA), 1 (AA), and 0 (aa):
	p = colMeans(X)/2

	# Mask for filtering markers:
	mask = p > cutoff 

	# Filtering markers:
	X = X[,mask]

	# Return processed matrix:
	return(X)
}

# Function to fix negative eigenvalues from the kinship matrix (K):
spec_dec <-function(K) {

  # Eigenvalue decomposition of the matrix:	
  E=eigen(K)

  # Creates a diagonal matrix for receiving the eigenvalues:
  Dg= diag(E$values)

  # Identifies eigenvalues lower then a small constant and pick the lowest one and add a fraction of it:
  for (i in 1:nrow(Dg)) {
    if (Dg[i,i]<1e-4) {
      Dg[i,i]=Dg[(i-1),(i-1)]-0.01*Dg[(i-1),(i-1)]
    }
  }

  # Creating a matrix of the eigen vectors:
  C=matrix(E$vectors,nrow(K),ncol(K))

  # Reconstructing the marker matrix:
  K = C%*%Dg%*%t(C)

  # Return the transformed matrix:
  return(K)

}


#-------------------------------------Subset the information from flags--------------------------------------#

# # Subset the arguments:
# args=commandArgs(trailingOnly = TRUE)
 
# # Get the file names:
# y = args[1]
# X = args[2]

# # Set the model:
# model = args[3]

# # Directory of the data:
# dir_in = args[4]

# # Directory of the project:
# dir_proj = args[5]

# # Directory where outputs will be saved:
# dir_out = args[6]

#******* Temp:
y = "y_cv1_height_k0_trn.csv"
X = "x_cv1_height_k0_trn.csv"
model = "MTiLM-0~6"
dir_in = "/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"
dir_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"
dir_out = "/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/MTiLM/cv1/height/k0"

# Load asreml:
setwd(paste0(dir_proj, 'asreml'))
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

#-----------------------------------------------Load data----------------------------------------------------#

# Load data:
if (str_detect(model, 'MTi')) {

	# Set directory:
	setwd(dir_in)

	# Read adjusted means:
	y = fread(y, header=TRUE) %>% data.matrix
	rownames(y) = y[,1]
	y = y[,-1]
	
	# Set directory:
	setwd(str_split(dir_in, '/cross', simplify = TRUE)[1,1])

	# Read marker matrix:
	X = fread('M.csv', header=TRUE)[,-1] %>% data.matrix %>% t(.)

}

# Set directory:
setwd(paste0(str_split(dir_out, '/cross', simplify = TRUE)[1,1], '/first_step_analysis'))

# Load data frame with adjusted means:
df = fread('adjusted_means.csv', header=TRUE) %>% data.frame
rownames(df) = df[,1]
df = df[,-1]
df$id_gbs = df$id_gbs %>% as.factor %>% droplevels
df$trait = df$trait %>% as.factor %>% droplevels
df$dap = df$dap %>% as.factor %>% droplevels

#****** Subset data:
# subset = df$id %>% unique %>% sample(150)
# df = df[df$id_gbs %in% subset,]
# X = X[as.character(subset),]

# Filtering marker with MAF lager then:
# X = filter_maf(X, 0.01)

# Create the relationship matrix:
# A = Gmatrix(X)
A = A.mat(X-1)

# Fixing eigenvalues and inverting the matrix:
A_inv = solve(spec_dec(A))
rownames(A_inv) = rownames(X)
colnames(A_inv) = rownames(X)

#--------------------------------------Prepare data format for asreml----------------------------------------#

# Subset data to build a data frame for asreml:
df_asreml = df[df$trait=='height',]
df_asreml = df_asreml %>% droplevels

# Ordering data frame:
df_asreml = df_asreml[order(df_asreml$id_gbs),]
# df_asreml = df_asreml[order(df_asreml$dap),]

# Changing the order of the factors:
df_asreml$dap<- factor(df_asreml$dap, levels = c(as.character(seq(30,120,15))))

# # Get the sample correlation matrix:
# guess = df %>% filter(dap != 'NA') %>%
#  			   spread(key = dap, value=y_hat) %>%
#  			   select_if(is.numeric) %>%
#  			   select(as.character(seq(30, 120, by=15))) %>%
#  			   cor(.) %>%
#  			   round(., 2)

# # Subset just values used for guesses:
# mask = upper.tri(guess, diag=TRUE)
# guess = c(guess[mask])

#--------------------------------------------Load libraries--------------------------------------------------#

# Fit a multivariate linear mixed model:


toc = list()

tic = proc.time()

fit = asreml(y_hat ~ dap,
			 random = ~ us(dap, 2):giv(id_gbs),
			 rcov = ~ id_gbs:diag(dap),
			 na.method.Y = "include",
			 control = asreml.control(
			 ginverse = list(id_gbs=A_inv),
			 workspace = 150e+06,
			 maxiter = 200),
             data = df_asreml)

toc[['fa2_diag']] = proc.time() - tic




# Number of factor analytic components:
n_fa = 2

# Initialize list:
index = list()

# Get the indexes for builing the G matrix under fa2 structure:
mask = str_detect(rownames(summary(fit)$varcomp), ".fa")
index[['load']] = rownames(summary(fit)$varcomp)[mask]
mask = str_detect(rownames(summary(fit)$varcomp), ".var$")
index[['var']] = rownames(summary(fit)$varcomp)[mask]

# Creating loading and specific variances matrices:
L = matrix(summary(fit)$varcomp[index[['load']],]$component,nlevels(df_asreml$dap), n_fa)
rownames(L) = paste0('HT_', levels(df_asreml$dap))
Psi = diag(summary(fit)$varcomp[index[['var']],]$component)
rownames(Psi) = paste0('HT_', levels(df_asreml$dap))

# Building genetic variance-covariance matrix:
G = L %*% t(L) + Psi 
tmp = paste0('HT_', seq(30,120,15))
G = G[tmp,tmp]
print(round(G,2))

# Building correlation matrix:
G_cor = diag(diag(G)^-0.5) %*% G %*% diag(diag(G)^-0.5)
rownames(G_cor) = tmp
colnames(G_cor) = tmp
print(round(G_cor,2))

# Get the index to build the R matrix for ar1h structure:
mask = str_detect(rownames(summary(fit)$varcomp), "R!dap")
index[['var_res']] = rownames(summary(fit)$varcomp)[mask][-1]
mask = str_detect(rownames(summary(fit)$varcomp), "cor")
index[['cor_res']] = rownames(summary(fit)$varcomp)[mask]

# Get the residual var-cov structure under ar1h:
var_res = as.matrix(summary(fit)$varcomp[index[['var_res']],]$component)
rho_res = summary(fit)$varcomp[index[['cor_res']],]$component
H = abs(outer(1:nlevels(df_asreml$dap), 1:nlevels(df_asreml$dap), "-"))
R = (var_res %*% t(var_res)) * rho_res^H
rownames(R) = tmp
colnames(R) = tmp
print(round(R,2))

# Get the residual correlation matrix:
R_cor = diag(diag(R)^-0.5) %*% R %*% diag(diag(R)^-0.5)
rownames(R_cor) = tmp
colnames(R_cor) = tmp
print(round(R_cor,2))



fit = asreml(y_hat ~ dap,
			 random = ~ fa(dap, 2):giv(id_gbs),
			 rcov = ~ units:at(dap),
			 na.method.Y = "include",
			 control = asreml.control(
			 ginverse = list(id_gbs=A_inv),
			 workspace = 150e+06,
			 maxiter = 100),
             data = df_asreml)


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#----------------------------------Load R data and write into .csv format files------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Prefix of the directory of the project is in:
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
prefix_proj = "/data1/aafgarci/jhonathan/mtrait-proj/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"


# Setting directory:
setwd(paste0(prefix_out, "data"))

# Loading marker data:
load("sorghum_PstI_info.RData")
load("sorghum_PstI_snps012.RData")
load("sorghum_PstI_taxa.RData")

write.csv(snps012, file="gbs.txt")
write.csv(taxa, file="taxa.txt")
write.csv(snp_info, file="gbs_info.txt")


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#--------------------------------------Fits multivariate rrBLUP on asreml------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Set directory:
setwd(str_split(dir_in, '/cross', simplify = TRUE)[1,1])

# Load bin matrix:
W_bin = fread('W_bin.csv', header=TRUE) %>% data.frame
rownames(W_bin) = W_bin[,1]
W_bin = W_bin[,-1]


# Subset data to build a data frame for asreml:
df_asreml = df[df$trait=='height',]
df_asreml = df_asreml %>% droplevels

# Initialize matrix for extend the matrix rows:
W_bin_expanded = W_bin

# Extend the matrix rows:
for (i in 2:nlevels(df_asreml$dap)) {

	W_bin_expanded = rbind(W_bin_expanded, W_bin)

}

# Merging new data frame with markers with proper ordering for asreml:
df_asreml = cbind(df_asreml, data.matrix(W_bin_expanded)[df_asreml$id_gbs,])

# Ordering data frame:
df_asreml = df_asreml[order(df_asreml$id_gbs),]

# Fit a multivariate linear mixed model:
fit = asreml(y_hat ~ dap,
			 random=~grp(bins):us(dap),
			 rcov=~id_gbs:us(dap),
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#----------------------------Compute the probabilities (not sure with right)---------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Dictionary to receive the accuracy matrices:
prob_dict = dict()

# Compute correlation for the Bayesian network and Pleiotropic Bayesian Network model under CV2 scheme:
for k in model_set:
  for i in range(len(dap_group[:-1])):
    # Subset predictions and observations for probability computation:
    y_pred_tmp = y_pred_cv2[k + '_cv2_height_trained!on!dap:' + dap_group[i]]
    y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group[i]].y_hat
    # Compute probability across DAP measures:
    for j in range(len(dap_group)):
      # Conditional to compute probability just forward in time:
      if (j>i):
        # Subset indexes for subsetting the data:
        subset = df[df.dap==int(dap_group[j])].index
        # Get the number of selected individuals for 20% selection intensity:
        n_selected = int(y_obs_tmp[subset].size * 0.2)
        # Build the indexes for computing the coeficient of coincidence:
        rank_obs = np.argsort(y_obs_tmp[subset])[::-1][0:n_selected]
        # Vector for storing the indicators:
        ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
        # Build probability matrix for the Bayesian Network model:
        for sim in range(y_pred_tmp.shape[0]):
          # Get the indicator of which genotype is in the top 20% or not: 
          ind_vec.iloc[sim] = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1].isin(rank_obs)
        # Index to store probabilties into dictionary:
        index = k + '_' + dap_group[i] + '_' + dap_group[j]
        # Compute probability:
        prob_dict[index]=ind_vec.mean(axis=0)
        print('Model: {}, DAP_i: {}, DAP_j: {}'.format(k, dap_group[i], dap_group[j]))

# Compute correlation for the Dynamic Bayesian network model under CV2 scheme:
for i in range(len(dap_group1)):
  # Subset predictions for correlation computation:
  y_pred_tmp = y_pred_cv2['dbn_cv2_height_trained!on!dap:' + dap_group1[i]]
  y_obs_tmp = y_obs_cv2['cv2_height_for!trained!on:' + dap_group1[i]].y_hat
  for j in range(len(dap_group2)):    
    # Get the upper bound of the interval:
    upper = int(dap_group1[i].split('~')[1])
    # Conditional to compute correlation just forward in time:
    if (int(dap_group2[j])>upper):
      # Subset indexes for subsetting the data:
      subset = df[df.dap==int(dap_group2[j])].index
      # Get the number of selected individuals for 20% selection intensity:
      n_selected = int(y_obs_tmp[subset].size * 0.2)
      # Build the indexes for computing the coeficient of coincidence:
      rank_obs = np.argsort(y_obs_tmp[subset])[::-1][0:n_selected]
      # Vector for storing the indicators:
      ind_vec = pd.DataFrame(index=y_pred_tmp[subset].index, columns=y_pred_tmp[subset].columns)
      # Build probability matrix for the Bayesian Network model:
      for sim in range(y_pred_tmp.shape[0]):
        # Get the indicator of which genotype is in the top 20% or not: 
        ind_vec.iloc[sim] = np.argsort(y_pred_tmp[subset].iloc[sim])[::-1].isin(rank_obs)
      # Index to store probabilties into dictionary:
      index = 'dbn_' + dap_group1[i] + '_' + dap_group2[j]
      # Compute probability:
      prob_dict[index]=ind_vec.mean(axis=0)
      print('Model: dbn, DAP_i: {}, DAP_j: {}'.format(dap_group1[i], dap_group2[j]))

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#--------------------------------Splitting feature matrices by time points-----------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# Creating an empty dictionary to receive the feature matrices and response vectors:
X = dict()
y = dict()

# Building the feature matrix for the height under different DAP's:
index = ['loc', 'year', 'dap', 'id_gbs']
tmp = df.dap.drop_duplicates()[1::].tolist()
for i in tmp:
	# Index for subsetting the data:
	subset = (df.trait=='height') & (df.dap==i)
	# Creating the matrix with the design and covariates structure:
	X['height_' + str(int(i))] = pd.get_dummies(df.loc[subset, index])
	# Adding the bin matrix to the feature matrix:
	geno = pd.get_dummies(df.id_gbs[subset])
	X['height_' + str(int(i))] = pd.concat([X['height_' + str(int(i))], geno.dot(W_bin.loc[geno.columns.tolist()])], axis=1)
	# Removing rows of the missing entries from the feature matrix:
	X['height_' + str(int(i))] = X['height_' + str(int(i))][~(df.height[subset].isnull())]
	# Creating a variable to receive the response without the missing values:
	y['height_' + str(int(i))] = df.height[subset][~(df.height[subset].isnull())]
	# Printing shapes:
	print((X['height_' + str(int(i))]).shape)
	print(y['height_' + str(int(i))].shape)


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------Construction of bins----------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

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


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#----------------------------Subdivision of the height data into mini-batches--------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

if cv=="CV1":
	# Subsetting the full set of names of the inbred lines phenotyped for biomass:
	index_mbatch = df.id_gbs[df.trait=='height'].drop_duplicates()
	# Size of the mini-batch
	size_mbatch = 4
	# Splitting the list of names of the inbred lines into 4 sublists for indexing the mini-batches:
	index_mbatch = np.array_split(index_mbatch, size_mbatch)
	# Type of sets:
	tmp = ['trn', 'dev', 'tst']
	# Indexing the mini-batches for the height trait:
	for k in tmp:
		for i in range(size_mbatch):
			# Getting the positions on the height training set related to the mini-batch i:
			index = df.id_gbs.loc[index_cv['cv1_height_' + k]].isin(index_mbatch[i])
			# Indexing height values of the mini-batch i:
			X['cv1_height_' + 'mb_' + str(i) + '_' + k ] = X['cv1_height_' + k][index]
			y['cv1_height_' + 'mb_' + str(i) + '_' + k ] = y['cv1_height_' + k][index]
			index_cv['cv1_height_' + 'mb_' + str(i) + '_' + k]  = index_cv['cv1_height_' + k][index]
			# Printing shapes:
			X['cv1_height_' + 'mb_' + str(i) + '_' + k ].shape
			y['cv1_height_' + 'mb_' + str(i) + '_' + k ].shape
			# Saving data:
			X['cv1_height_' + 'mb_' + str(i) + '_' + k ].to_csv('x_cv1_height_' + 'mb_' + str(i) + '_' + k  + '.csv')
			pd.DataFrame(y['cv1_height_' + 'mb_' + str(i) + '_' + k ], index=index_cv['cv1_height_' + 'mb_' + str(i) + '_' + k ]).to_csv('y_cv1_height_' + 'mb_' + str(i) + '_' + k  + '.csv')


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#----------------------------------------Example of a python class-------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

class Kls(object):
    def __init__(self, data):
        self.data = data
 
    def printd(self):
        print(self.data)
 
ik1 = Kls('arun')
ik2 = Kls('seema')
 
ik1.printd()
ik2.printd()

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#----------------------------------------Test via ridge regression-------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#transform y_train to match the evaluation metric
y_train=y['trn'].transpose()

#create X_train and X_test
X_train=X['trn'].transpose()
X_test=X['dev'].transpose()

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# steps
steps = [('scaler', StandardScaler()),
         ('ridge', Ridge())]
# steps = [('scaler', StandardScaler()),
#          ('lasso', Lasso())]


# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {('ridge__alpha'):np.logspace(-4, 0, 100)}

# Create the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

#predict on train set
y_pred_train=cv.predict(X_train)

# Predict test set
y_pred_test=cv.predict(X_test)

# rmse on train set
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Root Mean Squared Error: {}".format(rmse))

# rmse on test set
rmse = np.sqrt(mean_squared_error(y['dev'].transpose(), y_pred_test))
print("Root Mean Squared Error: {}".format(rmse))

# Plotting:
y_tmp = dict()
y_tmp['trn'] = y_pred_train
y_tmp['dev'] = y_pred_test
plt.scatter(y['trn'], y_tmp['trn'], color='red')
plt.scatter(y['dev'], y_tmp['dev'], color='green')
plt.xlim(2.5, 4)
plt.ylim(2.5, 4)
plt.title('Observed vs predicted data')
plt.xlabel('Observed transcription binned values')
plt.ylabel("Predicted transcription binned values")
plt.show()


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#--------------------------------------To run the code on pystan---------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

import pystan as ps

# Transpose:
X['trn'] = X['trn'].transpose()
X['dev'] = X['dev'].transpose()
X['tst'] = X['tst'].transpose()
y['trn'] = y['trn'].transpose()
y['dev'] = y['dev'].transpose()
y['tst'] = y['tst'].transpose()

# Getting the features names prefix:
tmp = X['trn'].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = np.ones(y['trn'].shape) 

# # For subsetting for tests:
# subset1 = np.random.choice(range(X['trn'].shape[0]), size=100)
# subset2 = X['trn'].index[subset1]

# # Storing all the data into a dictionary for pystan:
# df_stan = dict(n_x = X['trn'].loc[subset2,:].shape[0],
# 			   p_x = X['trn'].shape[1],
# 			   p_i = np.max(index_x),
# 			   p_r = X['year'].shape[1],
# 			   phi = np.max(y['trn'][subset1])*10,
# 			   index_x = index_x,
# 			   X = X['trn'].loc[subset2,:],
# 			   X_r = X['year'].loc[subset2,:],
# 			   y = y['trn'][subset1].reshape((y['trn'][subset1].shape[0],)))

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['trn'].shape[0],
			   p_x = X['trn'].shape[1],
			   p_i = np.max(index_x),
			   p_r = X['year'].shape[1],
			   phi = np.max(y['trn'])*10,
			   index_x = index_x,
			   X = X['trn'],
			   X_r = X['year'],
			   y = y['trn'].flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['400'] = model.sampling(data=df_stan, chains=1, iter=400)

# Getting posterior means:
beta_mean = dict()
mu_mean = dict()
beta_mean['400'] = fit['400'].extract()['beta'].mean(axis=0)
mu_mean['400'] = fit['400'].extract()['mu'].mean(axis=0)

# Computing predictions for trn:
y_pred = dict()
y_pred['trn'] = mu_mean['400'] + X['trn'].dot(beta_mean['400'])

# Printing train rMSE errors:
rmse(y['trn'].flatten(), y_pred['trn'])

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev rMSE errors:
rmse(y['dev'].flatten(), y_pred['dev'])

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test rMSE errors:
rmse(y['tst'].flatten(), y_pred['tst'])

# Printing train pearsonr:
pearsonr(y['trn'].flatten(), y_pred['trn'])[0]

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev pearsonr:
pearsonr(y['dev'].flatten(), y_pred['dev'])[0]

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test pearsonr:
pearsonr(y['tst'].flatten(), y_pred['tst'])[0]

# Plots of the observed against the generated:
sns.set_style('whitegrid')
ax = sns.kdeplot(fit['400'].extract()['y_rep'].mean(axis=0), bw=0.5, label='1_400', shade=True)
ax = sns.kdeplot(y['trn'].flatten(), bw=0.5, label='obs', shade=True)
ax.set_title('Observed vs generated data (nchain_niter)')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.savefig(prefix_out + 'plots/' + 'biomass_iter_tunning_obs_gen' + '.pdf')
plt.clf()

# Plotting:
plt.scatter(y['trn'], y_pred['trn'], color='red')
plt.scatter(y['dev'], y_pred['dev'], color='green')
plt.scatter(y['tst'], y_pred['tst'], color='blue')
# plt.xlim(2.5, 4)
# plt.ylim(2.5, 4)
plt.title('Observed vs predicted data')
plt.xlabel('Observed transcription binned values')
plt.ylabel("Predicted transcription binned values")
plt.show()
