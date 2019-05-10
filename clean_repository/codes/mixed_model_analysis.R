
#-------------------------------------------Loading libraries------------------------------------------------#

# Libraries to manipulate data:
library(optparse)
library(data.table)
library(stringr)
library(magrittr)
library(EMMREML)
library(rrBLUP)
library(tidyr)


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Set the flags:
option_list = list(
  make_option(c("-y", "--y"), type="character", default=NULL, 
              help="Name of the file with the phenotypes", metavar="character"),
  make_option(c("-m", "--model"), type="character", default=NULL, 
              help="Specify the model", metavar="character"),
  make_option(c("-o", "--opath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs", metavar="character"),
  make_option(c("-c", "--cvpath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs of the step of cross-validation analysis", metavar="character")
) 

# Parse the arguments:
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

# Subset arguments:
# y = args$y
# model = args$model
# OUT_PATH = args$opath
# CV_OUT_PATH = args$opath

y = 'y_fcv_drymass_trn.csv&y_fcv-30~60_height_trn.csv'
model = 'MTr-GBLUP-0~2'
OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'
CV_OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~60/drymass-height'

#---------------------------------Define function for spectral decomposition---------------------------------#

# Function to fix negative eigenvalues from the kinship matrix (K):
spec_dec <-function(K) {

  # Eigenvalue decomposition of the matrix:	
  E=eigen(K)

  # Create a diagonal matrix for receiving the eigenvalues:
  Dg= diag(E$values)

  # Identify eigenvalues lower then a small constant and pick the lowest one:
  for (i in 1:nrow(Dg)) {
    if (Dg[i,i]<1e-4) {
      Dg[i,i]=Dg[(i-1),(i-1)]-0.01*Dg[(i-1),(i-1)]
    }
  }

  # Create a matrix of the eigen vectors:
  C=matrix(E$vectors,nrow(K),ncol(K))

  # Reconstruct the marker matrix:
  K = C%*%Dg%*%t(C)

  # Return the transformed matrix:
  return(K)

}


#-----------------------------------------------Load data----------------------------------------------------#

# Load train data just to get the indexes of the train, and test sets, used in the Bayesian Networks analysis:

# Set the directory:
setwd(paste0(OUT_PATH, '/processed_data'))

if (str_detect(model, 'MTr')) {

	# Subset file names per trait:
	y = str_split(y, "&", simplify = TRUE)[1,1]

}

# Read adjusted means:
y = fread(y, header=TRUE) %>% data.matrix
rownames(y) = y[,1]
y = y[,-1]
	
# Read the whole marker matrix:
M = fread('M.csv', header=TRUE)[,-1] %>% data.matrix %>% t(.)

# Load data frame with adjusted means to be processed and used on EMMREML for mixed model analysis:
df = fread('adjusted_means.csv', header=TRUE) %>% data.frame
rownames(df) = df[,1]
df = df[,-1]
df$id_gbs = df$id_gbs %>% as.factor %>% droplevels
df$trait = df$trait %>% as.factor %>% droplevels
df$dap = df$dap %>% as.factor %>% droplevels

# Get the indexes of train set:
index = names(y)

# Get the inbred lines ID for train set:
id_trn = df[index,]$id_gbs %>% as.character %>% unique

# Get the indexes of test set:
id_tst = df$id_gbs %>% as.character %>% unique %>% setdiff(., id_trn)  

#-------------------------------------Prepare data format for EMMREML----------------------------------------#

# Create the relationship matrix:
A = A.mat(M-1)
rownames(A) = rownames(M)
colnames(A) = rownames(M)

# Change the names of the columns:
colnames(df) = c("id", "y_hat", "trait", "dap")

# Conditional based on models:
if (str_detect(model, 'MTi')) {

	# Vector with the time points:
	time = seq(30, 120, 15)

	# Subset just time points used for training:
	upper = str_split(model, '~', simplify = TRUE)[1,2] %>% as.numeric

	# Change DAP class on the data frame:
	df$dap = df$dap %>% as.character %>% as.numeric

	# Subset data frame:
	df_tmp = df[!is.na(df$dap),]
	df_tmp = df_tmp[df_tmp$dap <= time[upper+1],]

	# Melting the data frame:
	df_melt = df_tmp %>% spread(key = dap, value=y_hat)

	# Subset the desired columns:
	df_melt = df_melt[,c('id', as.character(seq(30, time[upper+1], by=15)))]

	# Change row and column names:
	rownames(df_melt) = df_melt$id
	colnames(df_melt) = c('id_gbs', paste0('h', seq(30, time[upper+1], by=15)))

}
if (str_detect(model, 'MTr')) {

	# Substitute the NA for a drymass for melting all traits together:
	df$dap = df$dap %>% as.character
	mask = df$dap == 'NA'
	df[mask,]$dap = 'drymass'

	# Vector with the time points:
	time = seq(30, 120, 15)

	# Subset just time points used for training:
	upper = str_split(model, '~', simplify = TRUE)[1,2] %>% as.numeric

	# Subset columns:
	df_tmp = df[,colnames(df)!='trait']

	# Melting the data frame:
	df_melt = df_tmp %>% spread(key = dap, value=y_hat)
	
	# Subset the desired columns:
	df_melt = df_melt[,c('id','drymass', as.character(seq(30,120,by=15)))]

	# Select height data used for train:
	tmp = time[time <= time[upper+1]]
	df_melt = df_melt[, c('id','drymass', tmp)]
	rownames(df_melt) = df_melt$id
	colnames(df_melt) = c('id_gbs','dm', paste0('h', seq(30, time[upper+1], by=15)))
	mask = !is.na(df_melt$dm)
	df_melt = df_melt[mask,]
	mask = rownames(A) %in% df_melt$id_gbs
	A = A[mask, mask]

}

# Reordering data frame:
df_melt = df_melt[rownames(A),] %>% droplevels
rownames(df_melt) = 1:nrow(df_melt)

# Changing the order of the factors:
df_melt$id_gbs = factor(df_melt$id_gbs, levels = c(as.character(df_melt$id_gbs)))

# Subset only individuals phenotyped and genotyped:
df_melt = df_melt[df_melt$id_gbs %in% rownames(A),]

# Reoder the relationship matrix as in the data frame:
index = as.character(df_melt$id_gbs)
A = A[index, index]

# Build design matrix of fixed effects:
n = nrow(df_melt)
X = matrix(1,nrow=1,ncol=n)
colnames(X) = df_melt$id_gbs %>% as.character

# Build design matrix of random effects:
Z = model.matrix(~ -1 + df_melt$id_gbs)
rownames(Z) = df_melt$id_gbs %>% as.character
colnames(Z) = df_melt$id_gbs %>% as.character

if (str_detect(model, 'MTi')) {

	# Preparing matrix with the response variable:
	select = paste0('h', seq(30,time[upper+1],15))
	Y = t(df_melt[, select])
	colnames(Y) = df_melt$id_gbs %>% as.character

}
if (str_detect(model, 'MTr')) {

	# Preparing matrix with the response variable:
	select = c('dm', paste0('h', seq(30,time[upper+1],15)))
	Y = t(df_melt[, select])
	colnames(Y) = df_melt$id_gbs %>% as.character

	# Update train and test set ID:
	id_trn = id_trn[id_trn %in% df_melt$id_gbs]
	id_tst = id_tst[id_tst %in% df_melt$id_gbs]

}

# Fit the multivariate linear mixed model:
out <- emmremlMultivariate(Y = Y[,id_trn],
			  			   X = X[,id_trn],
						   Z = Z[,id_trn],
						   K = spec_dec(A),
						   tolpar = 1e-4,
						   varBhat = FALSE, varGhat = FALSE, PEVGhat = FALSE, test = FALSE)

# Save output:
setwd(CV_OUT_PATH)
save.image("output_mixed_models.RData")
