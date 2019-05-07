
#-------------------------------------------Loading libraries------------------------------------------------#

# Import python libraries:
library(optparse)
library(data.table)
library(magrittr)
library(lme4)
library(stringr)


#-----------------------------------------Adding flags to the code-------------------------------------------#

# Set the flags:
option_list = list(
  make_option(c("-o", "--opath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs", metavar="character")
) 

# Parse the arguments:
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

# Subset arguments:
OUT_PATH = args$opath
# OUT_PATH = '/workdir/jp2476/output_sorghum-multi-trait'
# OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'

#---------------------------------------------Loading data---------------------------------------------------#

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

# # For mike:

# for (j in unique(as.character(df$env))) {

# 	subset=list()
# 	for (i in 1:16) {
# 		subset[[paste0('block_', i)]] = df[df$block==i & df$env == j & df$trait=='DM',]$id_gbs %>% as.character
# 	}

# 	print(paste0('Common checks for environment ', j, ':'))
# 	print(Reduce(intersect, subset))


# }

# for (d in unique(df$dap)[-1]) {
	
# 	for (j in unique(as.character(df$env))) {

# 		subset=list()
# 		for (i in 1:16) {
# 			subset[[paste0('block_', i)]] = df[df$block==i & df$env == j & df$trait=='PH' & df$dap==d,]$id_gbs %>% as.character
# 		}

# 		print(paste0('Common checks for environment ', j, ' and DAP ', d, ':'))
# 		print(Reduce(intersect, subset))


# 	}

# }


#------------------------------------------Biomass data analysis---------------------------------------------#

# Initialize a list to receive the first stage results:
fit = list()
g = list()

# Index for mapping the results:
index = "drymass"

# Subset part of the data set: 
df_tmp = df[df$trait == "DM"]

# Drop levels not present in the data subset:
df_tmp$id_gbs = df_tmp$id_gbs %>% droplevels
df_tmp$block = df_tmp$block %>% droplevels
df_tmp$env = df_tmp$env %>% droplevels

# Fit the model:
fit[[index]] = lmer(drymass ~ 1 + id_gbs + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

# Store the corrected mean:
g[[index]] = fixef(fit[[index]])[1] + fixef(fit[[index]])[-1]


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

	# Fitting the model:
	fit[[index]] = lmer(height ~ 1 + id_gbs + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

	# Storing the corrected mean:
	g[[index]] = fixef(fit[[index]])[1] + fixef(fit[[index]])[-1]

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

# Initialize a new data frame:
df_tmp = data.frame()

# Transform list into data frame:
for (i in names(g)) {

	if (i=='drymass') {
		 df_tmp = data.frame(id_gbs = str_split(names(g[[i]]), pattern="gbs", simplify = TRUE)[,2],
		 					 y_hat = unname(g[[i]]),
		 					 trait = rep('drymass', length(g[[i]])),
		 					 dap = rep("NA", length(g[[i]])))
    	 df_updated = df_tmp
	}
	if (i!='drymass') {
		 df_tmp = data.frame(id_gbs = str_split(names(g[[i]]), pattern="gbs", simplify = TRUE)[,2],
		 					 y_hat = unname(g[[i]]),
		 					 trait = rep('height', length(g[[i]])),
		 					 dap = rep(str_split(i, pattern="_", simplify = TRUE)[,2], length(g[[i]])))
		 df_updated = rbind(df_updated, df_tmp)
	}

}


#---------------------------------------------Saving outputs-------------------------------------------------#

# Set directory:
setwd(paste0(OUT_PATH, '/processed_data'))

# Save adjusted means:
write.csv(df_updated, file=paste0("adjusted_means.csv"))
