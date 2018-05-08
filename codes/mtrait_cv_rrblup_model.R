

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


#----------------------------------------------Loading data--------------------------------------------------#

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

# Loading full marker matrix:
setwd(paste0(prefix_out, 'data'))
M = t(fread('M.csv', header=TRUE)[,-1])




















