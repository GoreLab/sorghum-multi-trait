

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)
library(rrBLUP)
library(EMMREML)

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
# y = "y_cv1_height_k0_trn.csv"
y = "y_cv1_drymass_k0_trn.csv&y_cv1_height_k0_trn.csv"
# X = "x_cv1_height_k0_trn.csv"
X = "x_cv1_drymass_k0_trn.csv&x_cv1_height_k0_trn.csv"
model = "MTrLM-0~6"
dir_in = "/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"
dir_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"
dir_out = "/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/MTiLM/cv1/height/k0"

#-----------------------------------------------Load data----------------------------------------------------#

# Load train data just to get the indexes of the train, and test sets, used in the Bayesian Networks analysis:
if (str_detect(model, 'MTi')) {

	# Set directory:
	setwd(dir_in)

	# Read adjusted means:
	y = fread(y, header=TRUE) %>% data.matrix
	rownames(y) = y[,1]
	y = y[,-1]
	
}
if (str_detect(model, 'MTr')) {

	# Subset file names per trait:
	y_0 = str_split(y, "&", simplify = TRUE)[1,1]
	y_1 = str_split(y, "&", simplify = TRUE)[1,2]

	# Set directory:
	setwd(dir_in)

	# Read adjusted means:
	y_0 = fread(y_0, header=TRUE) %>% data.matrix
	rownames(y_0) = y_0[,1]
	y_0 = y_0[,-1]

	y_1 = fread(y_1, header=TRUE) %>% data.matrix
	rownames(y_1) = y_1[,1]
	y_1 = y_1[,-1]
	
}

# Set directory:
setwd(str_split(dir_in, '/cross', simplify = TRUE)[1,1])

# Read the whole marker matrix:
M = fread('M.csv', header=TRUE)[,-1] %>% data.matrix %>% t(.)

# Set directory:
setwd(paste0(str_split(dir_out, '/cross', simplify = TRUE)[1,1], '/first_step_analysis'))

# Load data frame with adjusted means to be processed and used on EMMREML for mixed model analysis:
df = fread('adjusted_means.csv', header=TRUE) %>% data.frame
rownames(df) = df[,1]
df = df[,-1]
df$id_gbs = df$id_gbs %>% as.factor %>% droplevels
df$trait = df$trait %>% as.factor %>% droplevels
df$dap = df$dap %>% as.factor %>% droplevels

# Get the indexes of train set:
if (str_detect(model, 'MTi')) {
	index = names(y)
}
if (str_detect(model, 'MTr')) {
	index = names(y_0)
}

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
colnames(df) = c('id', 'trait', 'dap', 'y_hat')

# Conditional based on models:
if (str_detect(model, 'MTi')) {

	# Melting the data frame:
	df_melt = df %>% filter(dap != 'NA') %>%
	 		      	 spread(key = dap, value=y_hat)

	# Subset the desired columns:
	df_melt = df_melt[,c('id', as.character(seq(30,120,by=15)))]

	# Change row and column names:
	rownames(df_melt) = df_melt$id
	colnames(df_melt) = c('id_gbs', paste0('h', seq(30,120,by=15)))

}
if (str_detect(model, 'MTr')) {

	# Substitute the NA for a drymass for melting all traits together:
	df$dap = df$dap %>% as.character
	mask = df$dap == 'NA'
	df[mask,]$dap = 'drymass'

	# Melting the data frame:
	df_melt = df %>% select(-trait) %>%
	 		      	 spread(key = dap, value=y_hat)
	 		      	 
	# Eliminate height phenotypes from only 11 inbred lines missing in the drymass for balance:
	mask = !is.na(df_melt$drymass)
	df_melt = df_melt[mask,]
	mask = rownames(A)[rownames(A) %in% df_melt$id_gbs]
	A = A[mask,]

	# Subset the desired columns:
	df_melt = df_melt[,c('id','drymass', as.character(seq(30,120,by=15)))]

	# Change row and column names:
	rownames(df_melt) = df_melt$id
	colnames(df_melt) = c('id_gbs','dm', paste0('h', seq(30,120,by=15)))

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

# Preparing matrix with the response variable:
select = paste0('h', seq(30,120,15))
Y = t(df_melt[, select])
colnames(Y) = df_melt$id_gbs %>% as.character

# Initialize list to store run time:
toc = list()

# Counting time:
tic = proc.time()

# Fit the multivariate linear mixed model:
out <- emmremlMultivariate(Y = Y[,id_trn],
			  			   X = X[,id_trn],
						   Z = Z[,id_trn],
						   K = spec_dec(A),
						   tolpar = 1e-4,
						   varBhat = FALSE, varGhat = FALSE, PEVGhat = FALSE, test = FALSE)

# Final time:
toc = proc.time() - tic

# Initialize list:
acc_yhat_lst = list()
G_cor_lst = list()
R_cor_lst = list()

# Naming the columns of the genomic estimated breeding values output:
colnames(out$Gpred) = as.character(df_melt$id_gbs)

# Accuracies between train and test set across different time points (20% unbalancing):
acc_yhat_lst[["with_spec_dec"]] = cor(t(Y[,id_tst]), t(out$Gpred[,id_tst]))

# Genetic correlation between time points:
G_cor_lst[["with_spec_dec"]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

# Residual correlation between time points:
R_cor_lst[["with_spec_dec"]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)
