
#--------------------------------------------Load libraries--------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)
library(rrBLUP)

# Library to run linear mixed model analysis:
library(sommer)

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


#--------------------------------------Prepare data format for sommer----------------------------------------#

# Change the names of the columns:
colnames(df) = c('id', 'trait', 'dap', 'y_hat')

# Melting the data frame:
df_melt = df %>% filter(dap != 'NA') %>%
 		      	 spread(key = dap, value=y_hat)

# Subset the desired columns:
df_melt = df_melt[,c('id', as.character(seq(30,120,by=15)))]

# Change row and column names:
rownames(df_melt) = df_melt$id
colnames(df_melt) = c('id_gbs', paste0('HT', seq(30,120,by=15)))

# Subset just the phenotyped and genotyped individuals:
X = X[intersect(rownames(X), rownames(df_melt)),]

# Create the relationship matrix:
A = A.mat(X-1)

# Keep the data frame with the same order as the relationship matrix:
df_melt = df_melt[rownames(A),]


df_melt_markers = cbind(df_melt, (X-1)[,1:100])


df_new = data.frame(cbind(df_melt[,'HT120'], (X-1)[,1:100]))
colnames(df_new) <- c('y', 1:(ncol(df_new)-1))
df_new$y = as.numeric(df_new$y)



fit = asreml(fixed = cbind(HT30,HT45,HT60,HT75,HT90,HT105,HT120)~-1,
			 random=~us(trait):grp(genotypes),
             rcov=~diag(trait):units,
             group=list(genotypes=9:ncol(df_melt_markers)),
             maxiter=100,
             data=df_melt_markers)


fit = asreml(HT120 ~ 1,
			 random=~grp(genotype),
             group=list(genotype=9:20),
             data=df_melt_markers)

model1 <- asreml(y ~ 1, random = ~ grp(genotype), 
                 group = list(genotype = 1:100), data = df_new) 