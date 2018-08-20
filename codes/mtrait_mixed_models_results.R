
#-------------------------------------------Loading libraries------------------------------------------------#

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


#-------------------------------------------Load outputs for CV1----------------------------------------------#

# Directories:
prefix_proj = '/workdir/jp2476/repo/sorghum-multi-trait/'
prefix_out = '/workdir/jp2476/repo/resul_mtrait-proj/'

# Types of cross-validation:
cv_type = paste0('k', 0:4)

# Initialize list:
G_cor_lst = list()
R_cor_lst = list()
y_pred_lst = list()

# Create a variable to store part of the directory:
dir_partial = '/outputs/cross_validation/MTiLM/cv1/height/'

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(prefix_out, dir_partial , cv_type[i]))

  # Load the entire workspace:
  load("output.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Broadcasting the mean:
  mu = matrix(rep(out$Bhat,each=ncol(out$Gpred[,id_tst])), ncol=ncol(out$Gpred[,id_tst]), byrow=TRUE)

  # Subset predictions:
  y_pred_tmp = t(mu) + t(out$Gpred[,id_tst])

  if (i==1) {

    # Subset predictions:
    y_pred_lst[['MTiLM']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTiLM']] = rbind(y_pred_lst[['MTiLM']], y_pred_tmp)

  }

  # Genetic correlation between time points:
  G_cor_lst[['MTiLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

  # Residual correlation between time points:
  R_cor_lst[['MTiLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)

  print(paste0("We are at the k: ", i))

}

# Initialize a list:
acc_lst = list()

# Saving predictive accuracy of the CV1 for the MTiLM:
index = df_melt$id_gbs %>% droplevels %>% as.character
rownames(df_melt) = index
acc_lst[['MTiLM']][['cv1']] = diag(cor(y_pred_lst[['MTiLM']][index,], df_melt[index,-1]))

# Create a variable to store part of the directory:
dir_partial = '/outputs/cross_validation/MTrLM/cv1/drymass-height/'

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(prefix_out, dir_partial , cv_type[i]))

  # Load the entire workspace:
  load("output.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Broadcasting the mean:
  mu = matrix(rep(out$Bhat,each=ncol(out$Gpred[,id_tst])), ncol=ncol(out$Gpred[,id_tst]), byrow=TRUE)

  # Subset predictions:
  y_pred_tmp = t(mu) + t(out$Gpred[,id_tst])

  if (i==1) {

    # Subset predictions:
    y_pred_lst[['MTrLM']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTrLM']] = rbind(y_pred_lst[['MTrLM']], y_pred_tmp)

  }

  # Genetic correlation between time points:
  G_cor_lst[['MTrLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

  # Residual correlation between time points:
  R_cor_lst[['MTrLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)

  print(paste0("We are at the k: ", i))

}

# Saving predictive accuracy of the CV1 for the MTiLM:
index = df_melt$id_gbs %>% droplevels %>% as.character
rownames(df_melt) = index
acc_lst[['MTrLM']][['cv1']] = diag(cor(y_pred_lst[['MTrLM']][index,], df_melt[index,-1]))

# Print predictive accuracies:
acc_lst[['MTiLM']][['cv1']]
acc_lst[['MTrLM']][['cv1']]
