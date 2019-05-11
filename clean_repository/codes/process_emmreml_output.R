
#-------------------------------------------Loading libraries------------------------------------------------#

# Libraries to manipulate data:
library(optparse)
library(data.table)
library(stringr)
library(magrittr)
library(EMMREML)
library(tidyr)


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Set the flags:
option_list = list(
  make_option(c("-r", "--rpath"), type="character", default=NULL, 
              help="The path of the repository", metavar="character"),
  make_option(c("-o", "--opath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs", metavar="character"),
) 

# Parse the arguments:
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

# Subset arguments:
REPO_PATH = args$rpath
OUT_PATH = args$opath
# REPO_PATH = '/workdir/jp2476/sorghum-multi-trait'
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'


#--------------Load outputs, predict and compute accuracies from 5-fold cross-validation (cv5f)-------------------#

# Types of cross-validation:
cv_type = paste0('k', 0:4)

# Initialize list:
y_pred_lst = list()

# Create a variable to store part of the directory:
PARTIAL_PATH = '/cv/MTi-GBLUP/cv5f/height'

# Read cv5f outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(OUT_PATH, PARTIAL_PATH , '/', cv_type[i]))

  # Load the entire workspace:
  load("output_mixed_models.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Broadcasting the mean:
  mu = matrix(rep(out$Bhat,each=ncol(out$Gpred[,id_tst])), ncol=ncol(out$Gpred[,id_tst]), byrow=TRUE)

  # Subset predictions:
  y_pred_tmp = t(mu) + t(out$Gpred[,id_tst])

  if (i==1) {

    # Subset predictions:
    y_pred_lst[['MTi-GBLUP']][['cv5f']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTi-GBLUP']][['cv5f']] = rbind(y_pred_lst[['MTi-GBLUP']][['cv5f']], y_pred_tmp)

  }

  print(paste0("We are at the k: ", i))

}

# Initialize a list:
acc_lst = list()

# Saving predictive accuracy of the cv5f for the MTi-GBLUP:
index = df_melt$id_gbs %>% droplevels %>% as.character
acc_lst[['MTi-GBLUP']][['cv5f']] = diag(cor(y_pred_lst[['MTi-GBLUP']][['cv5f']][index,], t(Y)[index,]))

# Create a variable to store part of the directory:
PARTIAL_PATH = '/cv/MTr-GBLUP/cv5f/drymass-height/'

# Read cv5f outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(OUT_PATH, PARTIAL_PATH , '/', cv_type[i]))

  # Load the entire workspace:
  load("output_mixed_models.RData")

  # Naming the columns of the genomic estimated breeding values output:
  colnames(out$Gpred) = as.character(df_melt$id_gbs)
  rownames(out$Gpred) = rownames(Y)

  # Broadcasting the mean:
  mu = matrix(rep(out$Bhat,each=ncol(out$Gpred[,id_tst])), ncol=ncol(out$Gpred[,id_tst]), byrow=TRUE)

  # Subset predictions:
  y_pred_tmp = t(mu) + t(out$Gpred[,id_tst])

  if (i==1) {

    # Subset predictions:
    y_pred_lst[['MTr-GBLUP']][['cv5f']] = y_pred_tmp

  }
  else {

    # Subset predictions and update the prediction matrix:
    y_pred_lst[['MTr-GBLUP']][['cv5f']] = rbind(y_pred_lst[['MTr-GBLUP']][['cv5f']], y_pred_tmp)

  }

  print(paste0("We are at the k: ", i))

}

# Saving predictive accuracy of the cv5f for the Mr-GBLUP:
index = df_melt$id_gbs %>% droplevels %>% as.character
acc_lst[['MTr-GBLUP']][['cv5f']] = diag(cor(y_pred_lst[['MTr-GBLUP']][['cv5f']][index,], t(Y)[index,]))


#----------Load outputs, predict and compute accuracies from forward chaining cross-validation (fcv)--------------#

# Create a variable to store part of the directory:
PARTIAL_PATH = '/cv/MTi-GBLUP'

# Types of cross-validation:
cv_type = paste0('fcv-30~', seq(45,105,15))

# Time points to correlate:
dap = seq(30,120,15)

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(OUT_PATH, PARTIAL_PATH, '/', cv_type[i], '/height/'))

  # Load the entire workspace:
  load("output_mixed_models.RData")

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
  y_pred_lst[['MTi-GBLUP']][[cv_type[i]]] = mu + t(out$Gpred[mask,])

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
      index = colnames(y_pred_lst[['MTi-GBLUP']][[cv_type[i]]])

      # Compute predictive accuraces:
      acc_matrix[as.character(j), upper] = cor(c(y_pred_lst[['MTi-GBLUP']][[cv_type[i]]]), c(y_tmp[index]))

  }

  print(paste0("We are at the cross-validation: ", cv_type[i]))

}

# Saving predictive accuracy of the fcv for the MTi-GBLUP:
acc_lst[['MTi-GBLUP']][['fcv']] = acc_matrix[3:nrow(acc_matrix), 2:(nrow(acc_matrix)-1)]

# Create a variable to store part of the directory:
PARTIAL_PATH = '/cv/MTr-GBLUP'

# Types of cross-validation:
cv_type = paste0('fcv-30~', seq(45,105,15))

# Time points to correlate:
dap = seq(30,120,15)

# Read CV1 outputs from the Multiple Time Linear Mixed model:
for (i in 1:length(cv_type)) {

  # Set the directory:
  setwd(paste0(OUT_PATH, PARTIAL_PATH, '/', cv_type[i], '/drymass-height/'))

  # Load the entire workspace:
  load("output_mixed_models.RData")

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
  y_pred_lst[['MTr-GBLUP']][[cv_type[i]]] = mu + t(out$Gpred[mask,])

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
      subset = names(y_tmp) %in% colnames(y_pred_lst[['MTr-GBLUP']][[cv_type[i]]])
      y_tmp = y_tmp[subset]

      # Get inbred line names to correlate in the right pair:
      index = colnames(y_pred_lst[['MTr-GBLUP']][[cv_type[i]]])

      # Compute predictive accuraces:
      acc_matrix[as.character(j), upper] = cor(c(y_pred_lst[['MTr-GBLUP']][[cv_type[i]]]), c(y_tmp[index]))

  }

  print(paste0("We are at the cross-validation: ", cv_type[i]))

}

# Saving predictive accuracy of the fcv for the MTi-GBLUP:
acc_lst[['MTr-GBLUP']][['fcv']] = acc_matrix[3:nrow(acc_matrix), 2:(nrow(acc_matrix)-1)]

# Set of models to compute the coincidence index:
model_set = c('MTi-GBLUP', 'MTr-GBLUP')

# Set of upper time points:
upper_set = seq(45,105,15)

# Create a vector to store the coincidence indexes:
ci_tmp = rep(NA, length(upper_set))
names(ci_tmp) = paste0('fcv-30~', upper_set)

# Initialize a list to store results:
ci_lst = list()

for (m in model_set) {
  for (i in upper_set) {

    # Subset the expectation of plant height for the i^th DAP:
    yhat_height = y_pred_lst[[m]][[paste0('fcv-30~', i)]]
    tmp = yhat_height %>% colnames
    yhat_height = c(yhat_height)
    names(yhat_height) = tmp

    # Subset adjusted mean for drymass related:
    yhat_drymass = df[df$trait=='drymass',]$y_hat
    names(yhat_drymass) = df[df$trait=='drymass',]$id %>% as.character

    # Number of individuals selected to order genotypes:
    n_selected = as.integer(length(yhat_drymass) * 0.2)

    # Get the index of the top observed selected inbred lines for dry mass:
    top_lines_obs = yhat_drymass[order(yhat_drymass, decreasing=TRUE)][1:n_selected] %>% names

    # Get the index of the predicted to be best selected inbred lines for height:
    top_lines_pred = yhat_height[order(yhat_height, decreasing=TRUE)][1:n_selected] %>% names

    # Compute the coincidence index:
    ci_tmp[paste0('fcv-30~', i)] = mean(top_lines_pred %in% top_lines_obs)

  }

  # Store the coincidence indexes in a list:
  ci_lst[[m]] = ci_tmp

}

# Set directory:
setwd(paste0(REPO_PATH, "/clean_repository/tables"))

# Rename entries of the cv5f accuracy:
names(acc_lst[['MTi-GBLUP']][['cv5f']]) = paste0('PH_', seq(30,120,15))
names(acc_lst[['MTr-GBLUP']][['cv5f']]) = c('DB', paste0('PH_', seq(30,120,15)))

# Write accuracy files:
setwd(paste0(prefix_proj, 'plots/cv/heatplot'))
for (i in names(acc_lst[["MTi-GBLUP"]])) write.csv(acc_lst[["MTi-GBLUP"]][[i]], file=paste0("acc_MTi-GBLUP_", i, ".csv"))
for (i in names(acc_lst[["MTr-GBLUP"]])) write.csv(acc_lst[["MTr-GBLUP"]][[i]], file=paste0("acc_MTr-GBLUP_", i, ".csv"))

# Write coincidence indexes:
for (i in names(ci_lst)) write.csv(ci_lst[[i]], file=paste0('coincidence_index_', i, '.csv'))
