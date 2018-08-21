
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


#------------------------------------------Load outputs for CV1----------------------------------------------#

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
    y_pred_lst[['MTiLM']][['cv1']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTiLM']][['cv1']] = rbind(y_pred_lst[['MTiLM']][['cv1']], y_pred_tmp)

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
acc_lst[['MTiLM']][['cv1']] = diag(cor(y_pred_lst[['MTiLM']][['cv1']][index,], t(Y)[index,]))

# Print predictive accuracies:
acc_lst[['MTiLM']][['cv1']]

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
    y_pred_lst[['MTrLM']][['cv1']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTrLM']][['cv1']] = rbind(y_pred_lst[['MTrLM']][['cv1']], y_pred_tmp)

  }

  # Genetic correlation between time points:
  G_cor_lst[['MTrLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

  # Residual correlation between time points:
  R_cor_lst[['MTrLM']][['cv1']][[cv_type[i]]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)

  print(paste0("We are at the k: ", i))

}

# Saving predictive accuracy of the CV1 for the MTiLM:
index = df_melt$id_gbs %>% droplevels %>% as.character
acc_lst[['MTrLM']][['cv1']] = diag(cor(y_pred_lst[['MTrLM']][['cv1']][index,], t(Y)[index,]))

# Print predictive accuracies:
acc_lst[['MTiLM']][['cv1']]
acc_lst[['MTrLM']][['cv1']]


#------------------------------------------Load outputs for CV2----------------------------------------------#

# Create a variable to store part of the directory:
dir_partial = 'outputs/cross_validation/MTiLM/'

# Types of cross-validation:
cv_type = paste0('cv2-30~', seq(45,105,15))

# Time points to correlate:
dap = seq(30,120,15)

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(prefix_out, dir_partial , cv_type[i], '/height/'))

  # Load the entire workspace:
  load("output.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Get the upper bound of the time inverval used for training:
  upper = str_split(cv_type[i], '~', simplify = TRUE)[1,2] %>% as.character

  # Get a mask of the time points only used for training:
  mask = str_detect(rownames(out$Gpred), upper)

  # Broadcast the mean and subset just the one related to the last time point:
  mu = matrix(rep(out$Bhat[mask,],each=ncol(out$Gpred)), ncol=ncol(out$Gpred), byrow=TRUE) %>% data.matrix

  # Compute predictions:
  y_pred_lst[['MTiLM']][[cv_type[i]]] = mu + t(out$Gpred[mask,])

  # Initialize a matrix to receive the accuracies:
  if (i == 1) {

    acc_matrix = matrix(NA, length(dap), length(dap))
    rownames(acc_matrix) = dap
    colnames(acc_matrix) = dap

  }

  # Intervals to be subset to compute accuracies
  begin = which(dap==upper)+1
  end = length(dap)

  # Compute accuracies:
  for (j in dap[begin:end]){
      
      # Subset observde data:
      y_tmp = df[(df$trait=='height') & (df$dap==j),]$y_hat
      names(y_tmp) = df[(df$trait=='height') & (df$dap==j),]$id

      # Get inbred line names to correlate in the right pair:
      index = colnames(y_pred_lst[['MTiLM']][[cv_type[i]]])

      # Compute predictive accuraces:
      acc_matrix[as.character(j), upper] = cor(c(y_pred_lst[['MTiLM']][[cv_type[i]]]), c(y_tmp[index]))

  }

  # Genetic correlation between time points:
  G_cor_lst[['MTiLM']][[cv_type[i]]][[cv_type[i]]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

  # Residual correlation between time points:
  R_cor_lst[['MTiLM']][[cv_type[i]]][[cv_type[i]]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)

  print(paste0("We are at the cross-validation: ", cv_type[i]))


}

# Saving predictive accuracy of the CV2 for the MTiLM:
acc_lst[['MTiLM']][['cv2']] = acc_matrix[3:nrow(acc_matrix), 2:(nrow(acc_matrix)-1)]

# Print predictive accuracies:
acc_lst[['MTiLM']][['cv1']]
acc_lst[['MTrLM']][['cv1']]
acc_lst[['MTiLM']][['cv2']]

# Create a variable to store part of the directory:
dir_partial = 'outputs/cross_validation/MTrLM/'

# Types of cross-validation:
cv_type = paste0('cv2-30~', seq(45,105,15))

# Time points to correlate:
dap = seq(30,120,15)

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(prefix_out, dir_partial , cv_type[i], '/drymass-height/'))

  # Load the entire workspace:
  load("output.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Get the upper bound of the time inverval used for training:
  upper = str_split(cv_type[i], '~', simplify = TRUE)[1,2] %>% as.character

  # Get a mask of the time points only used for training:
  mask = str_detect(rownames(out$Gpred), upper)

  # Broadcast the mean and subset just the one related to the last time point:
  mu = matrix(rep(out$Bhat[mask,],each=ncol(out$Gpred)), ncol=ncol(out$Gpred), byrow=TRUE) %>% data.matrix

  # Compute predictions:
  y_pred_lst[['MTrLM']][[cv_type[i]]] = mu + t(out$Gpred[mask,])

  # Initialize a matrix to receive the accuracies:
  if (i == 1) {

    acc_matrix = matrix(NA, length(dap), length(dap))
    rownames(acc_matrix) = dap
    colnames(acc_matrix) = dap

  }

  # Intervals to be subset to compute accuracies
  begin = which(dap==upper)+1
  end = length(dap)

  # Compute accuracies:
  for (j in dap[begin:end]){

      # Subset observde data:
      y_tmp = df[(df$trait=='height') & (df$dap==j),]$y_hat
      names(y_tmp) = df[(df$trait=='height') & (df$dap==j),]$id

      # Subset of observed inbred lines for both drymass and height:
      subset = names(y_tmp) %in% colnames(y_pred_lst[['MTrLM']][[cv_type[i]]])
      y_tmp = y_tmp[subset]

      # Get inbred line names to correlate in the right pair:
      index = colnames(y_pred_lst[['MTrLM']][[cv_type[i]]])

      # Compute predictive accuraces:
      acc_matrix[as.character(j), upper] = cor(c(y_pred_lst[['MTrLM']][[cv_type[i]]]), c(y_tmp[index]))

  }

  # Genetic correlation between time points:
  G_cor_lst[['MTrLM']][[cv_type[i]]][[cv_type[i]]] = diag(diag(out$Vg)^-0.5) %*% out$Vg %*% diag(diag(out$Vg)^-0.5)

  # Residual correlation between time points:
  R_cor_lst[['MTrLM']][[cv_type[i]]][[cv_type[i]]] = diag(diag(out$Ve)^-0.5) %*% out$Ve %*% diag(diag(out$Ve)^-0.5)

  print(paste0("We are at the cross-validation: ", cv_type[i]))


}

# Saving predictive accuracy of the CV2 for the MTiLM:
acc_lst[['MTrLM']][['cv2']] = acc_matrix[3:nrow(acc_matrix), 2:(nrow(acc_matrix)-1)]

# Print predictive accuracies:
acc_lst[['MTiLM']][['cv1']]
acc_lst[['MTrLM']][['cv1']]
acc_lst[['MTiLM']][['cv2']]
acc_lst[['MTrLM']][['cv2']]

# Write accuracy files:
setwd(paste0(prefix_proj, 'plots/cv/heatplot'))
for (i in names(acc_lst[["MTiLM"]])) write.csv(acc_lst[["MTiLM"]][[i]], file=paste0("acc_MTiLM_", i, ".csv"))
for (i in names(acc_lst[["MTrLM"]])) write.csv(acc_lst[["MTrLM"]][[i]], file=paste0("acc_MTrLM_", i, ".csv"))


