
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
# OUT_PATH = args$opath
OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'
# OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'
ASREML_PATH = '/workdir/jp2476/asreml'

#----------------------------------------------Load data-----------------------------------------------------#

# Set the directory:
setwd(paste0(OUT_PATH, '/processed_data'))

# Load the data:
df = fread('df.csv', header=T)[,-1]

# Change classes:
df$id_gbs = df$id_gbs %>% as.factor
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
h2_pev = data.frame(trait=c('DM', paste0('PH_', unique(df$dap[!is.na(df$dap)]))), h2=NA)
h2_comp = data.frame(trait=c('DM', paste0('PH_', unique(df$dap[!is.na(df$dap)]))), n_env=NA, n_plot=NA, h2=NA, se=NA)


#######

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

	for (i in unique(as.character(df_tmp$id_gbs))) {

		# Subset the line:
		mask = df_tmp$id_gbs==i 

		# Count the number of environments and plots the line i was evaluated:
		n_env_tmp = c(length(unique(as.character(df_tmp[mask, ]$env))))
		n_plot_tmp = c(nrow(df_tmp[mask, ]))
		names(n_env_tmp) = i
		names(n_plot_tmp) = i

		# Stack numbers in the vector:
		if (i==unique(as.character(df_tmp$id_gbs))[1]) {

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

	print(h2_comp)

}



n_plot[n_plot>4]

# DM
> n_plot[n_plot>4]
NSL50601 NSL50748 PI148084 PI148089 PI276801 PI524948 PI525882 
      34       34       34       64       64       64       34 

# PH_30

data.frame(df[df$id_gbs=='PI276801' & df$trait=='PH' & df$dap=='30' & df$block=='1'])


tmp.loc[((tmp['id_gbs']=='PI276801') & (df['block']=='1')),]

1074   PI276801     1  EF   16    PH   30        NaN    32.5
1091   PI276801     1  FF   16    PH   30        NaN    37.5
2052   PI276801     1  EF   16    PH   30        NaN    32.5

tmp.loc[1074, 'year']
tmp.loc[2052, 'year']

tmp.loc[1074] == tmp.loc[2052]







df[3][(df[3].name2== 'PI276801') & (df[3].block == '1')]
df[2][(df[2].name2== 'PI276801') & (df[2]['plot'] == '16EF0001')]
df[2][(df[2].name2== 'PI276801') & (df[2]['plot'] == '16FF0091')]
df[2][(df[2].name2== 'PI276801') & (df[2]['h1'] == 41.0)]
df[2][(df[2].name2== 'PI276801') & (df[2]['h1'] == 32.5)]


#######


     id_gbs block loc year trait dap drymass   height   env
1  PI276801     1  EF   16    PH  30      NA 35.16667 EF_16
9  PI276801     1  EF   16    PH  30      NA 32.50000 EF_16*


     id_gbs block loc year trait dap drymass   height   env
1  PI276801     1  EF   16    PH  45      NA 78.80000 EF_16
9  PI276801     1  EF   16    PH  45      NA 77.50000 EF_16
	

df[0][(df[0].name2== 'PI276801') & (df[0].block == 1)]

df[2][(df[2].name2 == 'PI276801') & (df[2].name2 == 'PI276801')]




> data.frame(df[df$id_gbs=='PI276801' & df$trait=='PH' & df$dap=='30' & df$block=='1'])
    id_gbs block loc year trait dap drymass height   env
1 PI276801     1  EF   16    PH  30      NA   32.5 EF_16
2 PI276801     1  EF   16    PH  30      NA   41.0 EF_16
3 PI276801     1  FF   16    PH  30      NA   37.5 FF_16
4 PI276801     1  FF   16    PH  30      NA   26.5 FF_16
5 PI276801     1  EF   16    PH  30      NA   32.5 EF_16
6 PI276801     1  FF   16    PH  30      NA   37.5 FF_16
7 PI276801     1  EF   17    PH  30      NA   17.0 EF_17
8 PI276801     1  MW   17    PH  30      NA   27.0 MW_17


          plot    name1     name2    h1    h2     h3     h4     h5     h6     h7   trait year
1290  16EF0001  PRE0146  PI276801  32.5  77.5  207.0  287.0  351.0  387.0  417.0  height   16
1324  16FF0091  PRE0146  PI276801  37.5  69.5  138.0  263.0  319.0  391.0  408.0  height   16

          plot    name1     name2    h1    h2     h3     h4     h5     h6     h7   trait year
1291  16EF0011  PRE0146  PI276801  41.0  79.5  228.0  312.0  353.0  401.0  426.0  height   16
1294  16EF0144  PRE0146  PI276801  41.0  64.0  209.5  291.0  344.0  367.5  427.0  height   16

          plot    name1     name2    h1    h2     h3     h4     h5     h6     h7   trait year
1290  16EF0001  PRE0146  PI276801  32.5  77.5  207.0  287.0  351.0  387.0  417.0  height   16
1325  16FF0144  PRE0146  PI276801  32.5  65.5  145.0  270.5  313.0  355.0  406.0  height   16



df[0].loc[df[0]['plot']=='16EF0011',]
df[1].loc[df[1]['plot']=='16EF0011',]
df[2].loc[df[2]['plot']=='16EF0011',]
df[3].loc[df[3]['plot']=='16EF0011',]




df.loc[df['plot']=='16EF0011',]



#--------------------------------Load asreml and function for heritability-------------------------------------#

# Load asreml:
setwd(ASREML_PATH)
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)


# Function to compute heritability:
get_h2_pev = function(fit) {

	# Get the PEV:
	pred = predict(fit, classify = 'id_gbs', sed=TRUE)
	pev <- unname(pred$predictions$avsed[2]^2)
	
	# Compute heritability:
	h2 <- 1-(pev/(2*summary(fit, all=T)$varcomp['id_gbs!id_gbs.var',2]))

	return(h2)

}

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

#-----------------------------------------Biomass data analysis------------------------------------------------#



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
h2_pev[h2_pev$trait == 'DM', 'h2'] = round(get_h2_pev(fit[[index]]), 4)
h2_comp[h2_pev$trait == 'DM', 'h2'] = round(get_h2(fit[[index]]), 4)
 

