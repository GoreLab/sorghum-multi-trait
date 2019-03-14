
#-------------------------------------------Loading libraries------------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"
	
# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)

# Library for data analysis:
library(sjstats)

# Load asreml:
setwd(paste0(prefix_proj, 'asreml'))
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

# Project prefix:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"
	
# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)

install.packages(paste0(prefix_proj, 'asreml/asreml_3.0_R_jy-linux-intel64.tar.gz'), repos = NULL, type="source")

# Library for data analysis:
library(sjstats)

# Load asreml:
setwd(paste0(prefix_proj, 'asreml'))
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

# Load the package to compute the standard deviations of the heritability:
library(nadiv)

# Function to get the heritability and its standard deviation estimates with asreml v3.0:
get_h2 <- function(asreml_obj, n_env, n_rep) {

	# Re-create the output from a basic, univariate model in asreml-R: 
	asrMod <- list(gammas = asreml_obj$gammas, gammas.type = asreml_obj$gammas.type, ai = asreml_obj$ai)

	# Name objects:
	names(asrMod[[1]]) <- names(asrMod[[2]]) <- names(asreml_obj$gammas)

	# Compute the heritability and its standard deviation:
	#---Observation: V4/n_env is the var_GxE /#locations and V5/n_env*n_rep is the var_E /# locations * # blocks
	formula = eval(parse(text=paste0('h2 ~ V3 / (V1 + V2 + V3 + V4/', n_env, '+ V5/', n_env*n_rep, ')')))

	return(nadiv:::pin(asrMod, formula))

}

#-----------------------------------------------Loading data-------------------------------------------------#

# Setting the directory:
setwd(paste0(prefix_out, 'data'))

# Loading the data:
df = fread('df.csv', header=T)[,-1]

# Observing the classes of the data frame:
str(df)

# Changing classes:
df$id_gbs = df$id_gbs %>% as.factor
df$block = df$block %>% as.factor
df$loc = df$loc %>% as.factor
df$year = df$year %>% as.factor
df$trait = df$trait %>% as.factor
df$dap = df$dap %>% as.integer
df$drymass = df$drymass %>% as.numeric
df$height = df$height %>% as.numeric

# Getting the group levels of the day after plantting (DAP) measures:
dap_groups = df$dap %>% unique %>% na.omit %>% as.integer

# Creating a new column combing location and years:
df$env <- paste0(as.character(df$loc), "_",as.character(df$year))
df$env <- df$env %>% as.factor()

# Creating a list to receive the first stage results:
fit = list()
metrics = list()

#------------------------------------------Biomass data analysis---------------------------------------------#

# Index for mapping the results:
index = "drymass"

# Subsetting part of the data set: 
df_tmp = df[df$trait == "DM"]

# Dropping levels not present in the data subset:
df_tmp$id_gbs = df_tmp$id_gbs %>% droplevels
df_tmp$block = df_tmp$block %>% droplevels
df_tmp$env = df_tmp$env %>% droplevels

# Getting number of levels:
n_env = df_tmp$env %>% nlevels
n_rep = df_tmp$block %>% nlevels

# Fitting the model:
fit[[index]]  = asreml(drymass~ 1,
			 		   random = ~ id_gbs + env + block:env + id_gbs:env,
			 		   na.method.Y = "include",
			 		   control = asreml.control(
			 		   maxiter = 200),
             		   data = df_tmp)

# Compute the heritability and its standard deviation:
metrics[["drymass-h2"]] = get_h2(fit[[index]], n_env=n_env, n_rep=n_rep)

#------------------------------------------Height data analysis----------------------------------------------#

for (i in dap_groups) {

	# Index for mapping the results:
	index = "height_" %>% paste0(i)

	# Subsetting part of the data set: 
	df_tmp = df[(df$trait == "PH" & df$dap == i)]  

	# Dropping levels not present in the data subset:
	df_tmp$id_gbs = df_tmp$id_gbs %>% droplevels
	df_tmp$block = df_tmp$block %>% droplevels
	df_tmp$env = df_tmp$env %>% droplevels

	# Getting number of levels:
	n_env = df_tmp$env %>% nlevels
	n_rep = df_tmp$block %>% nlevels

	fit[[index]]  = asreml(height ~ 1,
				 		   random = ~ id_gbs + block:env + env + id_gbs:env,
				 		   na.method.Y = "include",
				 		   control = asreml.control(
				 		   maxiter = 200),
	             		   data = df_tmp)

	# Getting the metrics to evaluate model performance:
	metrics[[paste0(index, "-h2")]] = get_h2(fit[[index]], n_env=n_env, n_rep=n_rep)

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

#---------------------------------------------Saving outputs-------------------------------------------------#

# Transforming list into table:
h_table = qdapTools::list_df2df(metrics, col1='Trial')

# Setting directory:
setwd(paste0(prefix_out, "outputs/first_step_analysis"))

# Saving results:
write.csv(h_table, file="mtrait_first_step_analysis_heritability.txt")

# Saving RData:
save.image("mtrait_first_step_analysis_heritability.RData")

# # Loading data:
# setwd(paste0(prefix_out, "outputs/first_step_analysis"))
# load("mtrait_first_step_analysis.RData")
