

#-------------------------------------------Loading libraries------------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/thy6/sorghum-multi-trait/"

# Libraries:
library(data.table)

# Load asreml:
setwd(paste0(prefix_proj, 'asreml'))
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

# Manipulate strings:
library(stringr)

# Transform list into table:
library(qdapTools)

# Load the package to compute the standard deviations of the heritability:
library(nadiv)

# Function to get the heritability and its standard deviation estimates with asreml v3.0:
get_h2 <- function(asreml_obj, n_env, n_rep) {

	# Re-create the output from a basic, univariate model in asreml-R: 
	asrMod <- list(gammas = asreml_obj$gammas, gammas.type = asreml_obj$gammas.type, ai = asreml_obj$ai)

	# Name objects:
	names(asrMod[[1]]) <- names(asrMod[[2]]) <- names(asreml_obj$gammas)

	# Compute the heritability and its standard deviation:
	#---Observation: V2/n_env is the var_GxE /#locations and V3/n_env*n_rep is the var_E /# locations * # blocks
	formula = eval(parse(text=paste0('h2 ~ V1 / (V1 + V2/', n_env, '+ V3/', n_env*n_rep, ')')))

	return(nadiv:::pin(asrMod, formula))

}


#-----------------------------------------------Loading data-------------------------------------------------#

# Setting the directory:
setwd(paste0(prefix_proj, 'data'))

# Loading the data:
df = data.frame(fread('elite_GCMS_phenos.txt', header=T))blu[s]

# Print head:
print(df[1:10,1:10])

# Subset phenotype names
mask = str_detect(colnames(df), 'GC')
pheno_names = colnames(df)[mask]

# Temporal data frame:
df_tmp = data.frame(df)[, !mask]

# Initialize a list:
fit = list()
blups = list()
h2 = list()

for (i in pheno_names) {

	# Add the desired phenotype:
	df_tmp$y = df[ , i]

	# Fitting the model:
	fit[[i]]  = asreml(y ~ 1 + env + block:env,
				  random = ~ line + line:env,
				  na.method.Y = "include",
				  control = asreml.control(
				  maxiter = 200),
	              data = df_tmp)

	# Subset BLUPs:
	blups[[i]] = fit[[i]]$coefficients$random[1:length(unique(df$line))]

	# Get the heritability:
	h2[[i]] = get_h2(fit[[i]], n_env=length(unique(df$env)), n_rep = length(unique(df$block)))

}

# Transforming list into table:
blup_table = qdapTools::list_df2df(blups, col1='Trait')
h_table = qdapTools::list_df2df(h2, col1='Trait')
