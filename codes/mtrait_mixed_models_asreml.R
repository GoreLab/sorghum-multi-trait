
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
setwd(paste0(prefix_proj, 'asreml'))
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


# Set directory:
setwd(str_split(dir_in, '/cross', simplify = TRUE)[1,1])

# Load bin matrix:
W_bin = fread('W_bin.csv', header=TRUE) %>% data.frame
rownames(W_bin) = W_bin[,1]
W_bin = W_bin[,-1]


#--------------------------------------Prepare data format for asreml----------------------------------------#

# Subset data to build a data frame for asreml:
df_asreml = df[df$trait=='height',]
df_asreml = df_asreml %>% droplevels

# Initialize matrix for extend the matrix rows:
W_bin_expanded = W_bin[,1:100]

# Extend the matrix rows:
for (i in 2:nlevels(df_asreml$dap)) {

	W_bin_expanded = rbind(W_bin_expanded, W_bin[,1:100])

}

# Merging new data frame with markers with proper ordering for asreml:
df_asreml = cbind(df_asreml, data.matrix(W_bin_expanded)[df_asreml$id_gbs,])

# # Ordering data frame:
df_asreml = df_asreml[order(df_asreml$id_gbs),]
# df_asreml = df_asreml[order(df_asreml$dap),]

fit = asreml(y_hat ~ dap,
			 random=~grp(bins):us(dap),
			 rcov=~diag(dap):units,
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)

fit = asreml(y_hat ~ dap,
			 random=~us(dap):grp(bins),
			 rcov=~diag(dap):units,
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)

fit = asreml(y_hat ~ dap,
			 random=~us(dap):grp(bins),
			 rcov=~units:diag(dap),
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)

fit = asreml(y_hat ~ dap,
			 random=~grp(bins):us(dap),
			 rcov=~diag(dap):units,
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)



# Best guess (work on ordering):
fit = asreml(y_hat ~ dap,
			 random=~grp(bins):us(dap),
			 rcov=~diag(dap):id_gbs,
             group=list(bins=5:ncol(df_asreml)),
             data=df_asreml)










