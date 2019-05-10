
#-------------------------------------------Loading libraries------------------------------------------------#

# Import python libraries:
library(optparse)
library(data.table)
library(magrittr)
library(stringr)


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Set the flags:
option_list = list(
  make_option(c("-a", "--asremlpath"), type="character", default=NULL, 
              help="The path of the folder with asreml license", metavar="character")
  make_option(c("-o", "--opath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs", metavar="character")
) 

# Parse the arguments:
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

# Subset arguments:
OUT_PATH = args$opath
ASREML_PATH = args$asremlpath
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'
# OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'
# ASREML_PATH = '/workdir/jp2476/asreml'


#--------------------------------Load asreml and function for heritability-------------------------------------#

# Load asreml:
setwd(ASREML_PATH)
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

# Function to get the heritability and its standard deviation estimates with asreml v3.0:
get_h2 <- function(asreml_obj, n_env, n_plot) {

	# Re-create the output from a basic, univariate model in asreml-R: 
	asrMod <- list(gammas = asreml_obj$gammas, gammas.type = asreml_obj$gammas.type, ai = asreml_obj$ai)

	# Name objects:
	names(asrMod[[1]]) <- names(asrMod[[2]]) <- names(asreml_obj$gammas)

	# Compute the heritability and its standard deviation:
	#---Observation: V4/n_env is the var_GxE/#environment and V5/#plot is the var_E/#plot
	formula = eval(parse(text=paste0('h2 ~ V3 / (V3 + V4/', n_env, '+ V5/', n_plot, ')')))

	return(nadiv:::pin(asrMod, formula))

}


#----------------------------------------------Load data-----------------------------------------------------#

# Set the directory:
setwd(paste0(OUT_PATH, '/processed_data'))

# Load the data:
df = fread('df.csv', header=T)[,-1]

# Change classes:
df$id_gbs = df$id_gbs %>% as.factor
df$name2 = df$name2 %>% as.factor
df$block = df$block %>% as.factor
df$loc = df$loc %>% as.factor
df$year = df$year %>% as.factor
df$trait = df$trait %>% as.factor
df$dap = df$dap %>% as.integer
df$drymass = df$drymass %>% as.numeric
df$height = df$height %>% as.numeric

# Get the group levels of the day after plantting (DAP) measures:
dap_groups = df$dap %>% unique %>% na.omit %>% as.integer

# Create a new column combing location and years:
df$env <- paste0(as.character(df$loc), "_",as.character(df$year))
df$env <- df$env %>% as.factor()

# Creating a list to receive the first stage results:
fit = list()

# Create a table to receive the heritabilities:
h2_comp = data.frame(trait=c('DM', paste0('PH_', unique(df$dap[!is.na(df$dap)]))), n_env=NA, n_plot=NA, h2=NA, se=NA)


#--------Compute the harmonic means of the number of environments and plots the lines were evaluated---------#

# Compute the harmonic mean of the number of environments in which each line was observed
#-->  and the harmonic mean of the total number of plots in which each line was observed:
for (t in as.character(h2_comp$trait)) {

	# Mask to subset the data of the trait:
	if (t == 'DM') {
		mask = df$trait=='DM' & (!is.na(df$drymass))
	}
	if (str_detect(t, 'PH')) {
		mask = df$trait=='PH' & df$dap == str_split(t, pattern="_", simplify = TRUE)[,2] & (!is.na(df$height))
	}

	# Subset the data:
	df_tmp = df[mask]

	# Remove hybrids from the data:
	mask = !(df_tmp$name2 %in% c('Pacesetter', 'SPX'))
	df_tmp = df_tmp[mask]

	for (i in unique(as.character(df_tmp$name2))) {

		# Subset the line:
		mask = df_tmp$name2==i 

		# Count the number of environments and plots the line i was evaluated:
		n_env_tmp = c(length(unique(as.character(df_tmp[mask, ]$env))))
		n_plot_tmp = c(nrow(df_tmp[mask, ]))
		names(n_env_tmp) = i
		names(n_plot_tmp) = i

		# Stack numbers in the vector:
		if (i==unique(as.character(df_tmp$name2))[1]) {

			n_env = n_env_tmp
			n_plot = n_plot_tmp

		}
		else {

			n_env = c(n_env, n_env_tmp)
			n_plot = c(n_plot, n_plot_tmp)

		}

	}

	# Compute the harmonic mean of the number of environments and plots lines were evaluated:
	h2_comp[h2_comp$trait==t, 'n_env'] = 1/mean(1/n_env)
	h2_comp[h2_comp$trait==t, 'n_plot'] = 1/mean(1/n_plot)

}


#-----------------------------------------Biomass data analysis------------------------------------------------#

# Adding to the data frame dummy variables to discriminate diverse lines from hybrids:
df$d_hybrids = as.numeric((df$name2 %in% c('Pacesetter', 'SPX')))
df$d_lines = as.numeric(!(df$name2 %in% c('Pacesetter', 'SPX')))

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
fit[[index]]  = asreml(drymass~ 1 + id_gbs:d_hybrids,
			 		   random = ~ id_gbs:d_lines + env + block:env + id_gbs:d_lines:env,
			 		   na.method.Y = "include",
			 		   control = asreml.control(
			 		   maxiter = 200),
             		   data = df_tmp)

# Compute the heritability and its standard deviation:
h2_comp[h2_pev$trait == 'DM', c('h2','se')] = round(get_h2(fit[[index]], 
										  				   n_env=h2_comp$n_env[h2_pev$trait == 'DM'],
														   n_plot=h2_comp$n_plot[h2_pev$trait == 'DM']), 4)
 

#------------------------------------------Height data analysis------------------------------------------------#

for (index in as.character(h2_comp$trait)[-1]) {

	# Subsetting part of the data set:
	mask  = df$trait=='PH' & df$dap == str_split(index, pattern="_", simplify = TRUE)[,2] 
	df_tmp = df[mask]  

	# Dropping levels not present in the data subset:
	df_tmp$id_gbs = df_tmp$id_gbs %>% droplevels
	df_tmp$block = df_tmp$block %>% droplevels
	df_tmp$env = df_tmp$env %>% droplevels

	# Getting number of levels:
	n_env = df_tmp$env %>% nlevels
	n_rep = df_tmp$block %>% nlevels

	fit[[index]]  = asreml(height ~ 1 + id_gbs:d_hybrids,
				 		   random = ~ id_gbs:d_lines + block:env + env + id_gbs:d_lines:env,
				 		   na.method.Y = "include",
				 		   control = asreml.control(
				 		   maxiter = 200),
	             		   data = df_tmp)

	# Compute the heritability and its standard deviation:
	h2_comp[h2_pev$trait == index, c('h2','se')] = round(get_h2(fit[[index]], 
											  				   n_env=h2_comp$n_env[h2_pev$trait == index],
															   n_plot=h2_comp$n_plot[h2_pev$trait == index]), 4)

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

# Set the directory:
setwd(paste0(OUT_PATH, '/heritabilities'))

# Save adjusted means:
write.csv(h2_comp, file="heritabilities.csv")
