

#-------------------------------------------Loading libraries------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)

# Library for plots:
library(ggplot2)
library(ggpubr)

# Library for analysis:
library(rrBLUP)
library(lme4)
library(Metrics)

# To construct the Cockerham's model:
marker_model <- function(M) {

  # Computing allelic frequency:
  freq = colMeans(M)/2
  Fa = matrix(freq,nrow(M),ncol(M))
  
  # additive Cockerham's model:
  W = M-2*Fa

  return(W)
  
}

#-----------------------------------------------Loading data-------------------------------------------------#

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

# Loading the full data set:
setwd(paste0(prefix_out, 'data'))
df = fread('df.csv', header=T)[,-1]
rownames(df) = c(0, rownames(df)[-nrow(df)])

# Loading full marker matrix:
M = t(fread('M.csv', header=TRUE)[,-1]) - 1 

# Loading the bin matrix:
W_bin = data.frame(fread('W_bin.csv', header=TRUE))
rownames(W_bin) = W_bin[,1]
W_bin = data.matrix(W_bin[,-1])

# Building the matrix with the Cocherham's model:
W_full = marker_model((M+1))


#----------------------------------------Biomass two-stage analysis------------------------------------------#

# Indexing just the part of the data frame for training:
index = df$trait == 'biomass' & (!is.na(df[,'drymass']))
df_trn = data.frame(df)[index,]
rownames(df_trn) <- rownames(df)[index]

# First step analysis:
fit = list()
fit[['1st_biomass']] = lmer(drymass ~ 1 + id_gbs + (1|loc) + (1|year) + (1|id_gbs:loc) + (1|id_gbs:year), data=df_trn)

# Preparing output from the first stage to perform the second stage analysis:
blue = fixef(fit[['1st_biomass']])[-1]
index = str_split(names(blue), pattern='d_gbs',simplify=TRUE)[,2]
names(blue) <- index

# Creating indexes for cross-validation:
index_trn = sample(names(blue), round(length(blue)*0.7))
tmp = setdiff(names(blue), index_trn)
index_dev = sample(tmp, round(length(tmp)*0.5))
index_tst = setdiff(tmp, index_dev)

# Length of the sets:
length(index_trn)
length(index_dev)
length(index_tst)

# Fitting the GBLUP and rrBLUP model:
fit[['2st_cv1_biomass_trn_full']] = mixed.solve(y=blue[index_trn], Z=W_full[index_trn,])
fit[['2st_cv1_biomass_trn_bin']] = mixed.solve(y=blue[index_trn], Z=W_bin[index_trn,])

# Fitting the rrBLUP model:
y_pred = list()
y_pred[['trn_full']] = W_full[index_trn,] %*% fit[['2st_cv1_biomass_trn_full']]$u
y_pred[['dev_full']] = W_full[index_dev,] %*% fit[['2st_cv1_biomass_trn_full']]$u
y_pred[['tst_full']] = W_full[index_tst,] %*% fit[['2st_cv1_biomass_trn_full']]$u

y_pred[['trn_bin']] = W_bin[index_trn,] %*% fit[['2st_cv1_biomass_trn_bin']]$u
y_pred[['dev_bin']] = W_bin[index_dev,] %*% fit[['2st_cv1_biomass_trn_bin']]$u
y_pred[['tst_bin']] = W_bin[index_tst,] %*% fit[['2st_cv1_biomass_trn_bin']]$u

# Correlation:
cor_bin_full = matrix(NA,3,1)
cor_bin_full[1] = cor(y_pred[['trn_full']], y_pred[['trn_bin']])
cor_bin_full[2] = cor(y_pred[['dev_full']], y_pred[['dev_bin']])
cor_bin_full[3] = cor(y_pred[['tst_full']], y_pred[['tst_bin']])
rownames(cor_bin_full) <- c('trn', 'dev', 'tst')
colnames(cor_bin_full) <- c('cor_bin_all_marker')

# Accuracy:
cor_models <- matrix(NA,3,2)
cor_models[1,1] = cor(y_pred[['trn_full']], blue[index_trn])
cor_models[2,1] = cor(y_pred[['dev_full']], blue[index_dev])
cor_models[3,1] = cor(y_pred[['tst_full']], blue[index_tst])
cor_models[1,2] = cor(y_pred[['trn_bin']], blue[index_trn])
cor_models[2,2] = cor(y_pred[['dev_bin']], blue[index_dev])
cor_models[3,2] = cor(y_pred[['tst_bin']], blue[index_tst])
rownames(cor_models) <- c('trn', 'dev', 'tst')
colnames(cor_models) <- c('all_markers', 'bin')
print(round(cor_models,4))

# rMSE:
rmse_models <- matrix(NA,3,2)
rmse_models[1,1] = rmse(y_pred[['trn_full']], blue[index_trn])
rmse_models[2,1] = rmse(y_pred[['dev_full']], blue[index_dev])
rmse_models[3,1] = rmse(y_pred[['tst_full']], blue[index_tst])
rmse_models[1,2] = rmse(y_pred[['trn_bin']], blue[index_trn])
rmse_models[2,2] = rmse(y_pred[['dev_bin']], blue[index_dev])
rmse_models[3,2] = rmse(y_pred[['tst_bin']], blue[index_tst])
rownames(rmse_models) <- c('trn', 'dev', 'tst')
colnames(rmse_models) <- c('all_markers', 'bin')
print(round(rmse_models,4))

# Creating a list to store results:
metrics = list()
metrics[['cor_bin_full_biomass']] = cor_bin_full 
metrics[['cor_biomass']] = cor_models
metrics[['rmse_biomass']] = rmse_models


#-----------------------------------------Height two-stage analysis------------------------------------------#

# Indexing just the part of the data frame for training:
index = df$trait == 'height' & (!is.na(df$height))
df_trn = data.frame(df)[index,]
rownames(df_trn) <- rownames(df)[index]

# First step analysis:
fit[['1st_height']] = lmer(height ~ 1 + id_gbs + (1|loc) + (1|year) + (1|id_gbs:loc) + (1|id_gbs:year), data=df_trn)

# Preparing output from the first stage to perform the second stage analysis:
blue = fixef(fit[['1st_height']])[-1]
index = str_split(names(blue), pattern='d_gbs',simplify=TRUE)[,2]
names(blue) <- index

# Creating indexes for cross-validation:
index_trn = sample(names(blue), round(length(blue)*0.7))
tmp = setdiff(names(blue), index_trn)
index_dev = sample(tmp, round(length(tmp)*0.5))
index_tst = setdiff(tmp, index_dev)

# Length of the sets:
length(index_trn)
length(index_dev)
length(index_tst)

# Fitting the GBLUP and rrBLUP model:
fit[['2st_cv1_height_trn_full']] = mixed.solve(y=blue[index_trn], Z=W_full[index_trn,])
fit[['2st_cv1_height_trn_bin']] = mixed.solve(y=blue[index_trn], Z=W_bin[index_trn,])

# Fitting the rrBLUP model:
y_pred = list()
y_pred[['trn_full']] = W_full[index_trn,] %*% fit[['2st_cv1_height_trn_full']]$u
y_pred[['dev_full']] = W_full[index_dev,] %*% fit[['2st_cv1_height_trn_full']]$u
y_pred[['tst_full']] = W_full[index_tst,] %*% fit[['2st_cv1_height_trn_full']]$u

y_pred[['trn_bin']] = W_bin[index_trn,] %*% fit[['2st_cv1_height_trn_bin']]$u
y_pred[['dev_bin']] = W_bin[index_dev,] %*% fit[['2st_cv1_height_trn_bin']]$u
y_pred[['tst_bin']] = W_bin[index_tst,] %*% fit[['2st_cv1_height_trn_bin']]$u

# Correlation:
cor_bin_full = matrix(NA,3,1)
cor_bin_full[1] = cor(y_pred[['trn_full']], y_pred[['trn_bin']])
cor_bin_full[2] = cor(y_pred[['dev_full']], y_pred[['dev_bin']])
cor_bin_full[3] = cor(y_pred[['tst_full']], y_pred[['tst_bin']])
rownames(cor_bin_full) <- c('trn', 'dev', 'tst')
colnames(cor_bin_full) <- c('cor_bin_all_marker')

# Accuracy:
cor_models <- matrix(NA,3,2)
cor_models[1,1] = cor(y_pred[['trn_full']], blue[index_trn])
cor_models[2,1] = cor(y_pred[['dev_full']], blue[index_dev])
cor_models[3,1] = cor(y_pred[['tst_full']], blue[index_tst])
cor_models[1,2] = cor(y_pred[['trn_bin']], blue[index_trn])
cor_models[2,2] = cor(y_pred[['dev_bin']], blue[index_dev])
cor_models[3,2] = cor(y_pred[['tst_bin']], blue[index_tst])
rownames(cor_models) <- c('trn', 'dev', 'tst')
colnames(cor_models) <- c('all_markers', 'bin')
print(round(cor_models,4))

# rMSE:
rmse_models <- matrix(NA,3,2)
rmse_models[1,1] = rmse(y_pred[['trn_full']], blue[index_trn])
rmse_models[2,1] = rmse(y_pred[['dev_full']], blue[index_dev])
rmse_models[3,1] = rmse(y_pred[['tst_full']], blue[index_tst])
rmse_models[1,2] = rmse(y_pred[['trn_bin']], blue[index_trn])
rmse_models[2,2] = rmse(y_pred[['dev_bin']], blue[index_dev])
rmse_models[3,2] = rmse(y_pred[['tst_bin']], blue[index_tst])
rownames(rmse_models) <- c('trn', 'dev', 'tst')
colnames(rmse_models) <- c('all_markers', 'bin')
print(round(rmse_models,4))

# Creating a list to store results:
metrics[['cor_bin_full_height']] = cor_bin_full 
metrics[['cor_height']] = cor_models
metrics[['rmse_height']] = rmse_models


#--------------------------------------------One step approach-----------------------------------------------#
# Setting directory:
setwd(paste0(prefix_out, 'data/cross_validation/cv1'))

# Initialize lists:
y = list()
X = list()

# Loading phenotypic data:
cv_types = c('cv1_biomass_dev', 'cv1_biomass_trn', 'cv1_biomass_tst',
			 'cv1_height_dev', 'cv1_height_dev_mb_0', 'cv1_height_dev_mb_1', 'cv1_height_dev_mb_2', 'cv1_height_dev_mb_3',
			 'cv1_height_trn', 'cv1_height_trn_mb_0', 'cv1_height_trn_mb_1', 'cv1_height_trn_mb_2', 'cv1_height_trn_mb_3',
			 'cv1_height_tst', 'cv1_height_tst_mb_0', 'cv1_height_tst_mb_1', 'cv1_height_tst_mb_2', 'cv1_height_tst_mb_3')

# Loading phenotypic data:
for (i in 1:length(cv_types)) {

	y_tmp = data.matrix(fread(paste0('y_',cv_types[i],'.csv'), header=TRUE))
	rownames(y_tmp) = y_tmp[,1]
	y[[cv_types[i]]] = y_tmp[,-1]

}

# Loading feature matrices:
for (i in 1:length(cv_types)) {

	X_tmp = data.matrix(fread(paste0('x_',cv_types[i],'.csv'), header=TRUE))
	rownames(X_tmp) = X_tmp[,1]
	X[[cv_types[i]]] = X_tmp[,-1]

}

# Indexing cv1_biomass_trn data:
y_tmp = y[['cv1_biomass_trn']] 
X_tmp = X[['cv1_biomass_trn']]

# Splitting the feature matrix into fixed effects incidence matrices:
index = str_detect(colnames(X_tmp),paste(c('loc','year'),collapse='|'))
X_fixed_tmp = X_tmp[, index]

# Splitting the feature matrix into random effects incidence matrices:
index = str_detect(colnames(X_tmp),'id_gbs')
X_random_tmp = X_tmp[, index]
colnames(X_random_tmp) = str_split(colnames(X_random_tmp), pattern="_gbs_", simplify=TRUE)[,2]

# Checking if the ordering is ok of the design matrix and relationship matrix:
all(colnames(X_random_tmp) == colnames(A))

# Checking if the ordering is ok of the design matrices and response vector:
all(names(y_tmp) == rownames(X_fixed_tmp))
all(names(y_tmp) == rownames(X_random_tmp))


fit[['2st_cv1_biomass_trn']] = mixed.solve(y=y_tmp, X=X_fixed_tmp, Z=X_random_tmp, K=A)




#---------------------------------------------------Junk-----------------------------------------------------#


# # Summary of the relationship coefficients:
# tmp = A
# diag(tmp) = NA
# tmp[!is.na(tmp)] %>% summary() %>% print()

# # Building manual relationship matrix:
# p = colMeans(M+1)/2
# tmp = tcrossprod((M+1) - matrix(2*p, nrow(M), ncol(M), byrow=TRUE) )
# A2 = tmp / diag(tmp)



