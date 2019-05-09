
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
              help="The path of the folder with asreml license", metavar="character"),
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

#---------------------------------------------Load asreml----------------------------------------------------#

# Load asreml:
setwd(ASREML_PATH)
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

#---------------------------------------------Loading data---------------------------------------------------#

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


#------------------------------------------Biomass data analysis---------------------------------------------#

# Initialize a list to receive the first stage results:
fit = list()
g = list()

# Index for mapping the results:
index = "drymass"

# Subset part of the data set: 
df_tmp = df[df$trait == "DM"]

# Drop levels not present in the data subset:
df_tmp$name2 = df_tmp$name2 %>% droplevels
df_tmp$block = df_tmp$block %>% droplevels
df_tmp$env = df_tmp$env %>% droplevels

fit[[index]] = asreml(drymass~ 1 + name2,
			 		  random = ~ block:env + env + name2:env,
			 		  na.method.Y = "include",
			 		  control = asreml.control(
			 		  maxiter = 200),
             		  data = df_tmp)


# Extract fixed effects:
fixed_eff = summary(fit[[index]], all=T)$coef.fi[,1]

# Store the corrected mean:
g[[index]] = fixed_eff[length(fixed_eff)] + fixed_eff[-length(fixed_eff)]


#------------------------------------------Height data analysis----------------------------------------------#

for (i in dap_groups) {

	# Index for mapping the results:
	index = "height_" %>% paste0(i)

	# Subsetting part of the data set: 
	df_tmp = df[(df$trait == "PH" & df$dap == i)]  

	# Dropping levels not present in the data subset:
	df_tmp$name2 = df_tmp$name2 %>% droplevels
	df_tmp$block = df_tmp$block %>% droplevels
	df_tmp$env = df_tmp$env %>% droplevels

	fit[[index]] = asreml(height~ 1 + name2,
				 		  random = ~ block:env + env + name2:env,
				 		  na.method.Y = "include",
				 		  control = asreml.control(
				 		  maxiter = 200),
	             		  data = df_tmp)


	# Extract fixed effects:
	fixed_eff = summary(fit[[index]], all=T)$coef.fi[,1]

	# Store the corrected mean:
	g[[index]] = fixed_eff[length(fixed_eff)] + fixed_eff[-length(fixed_eff)]

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

# Initialize a new data frame:
df_tmp = data.frame()

# Transform list into data frame:
for (i in names(g)) {

	if (i=='drymass') {
		 df_tmp = data.frame(name2 = str_split(names(g[[i]]), pattern="name2_", simplify = TRUE)[,2],
		 					 y_hat = unname(g[[i]]),
		 					 trait = rep('drymass', length(g[[i]])),
		 					 dap = rep("NA", length(g[[i]])))
    	 df_updated = df_tmp
	}
	if (i!='drymass') {
		 df_tmp = data.frame(name2 = str_split(names(g[[i]]), pattern="name2_", simplify = TRUE)[,2],
		 					 y_hat = unname(g[[i]]),
		 					 trait = rep('height', length(g[[i]])),
		 					 dap = rep(str_split(i, pattern="_", simplify = TRUE)[,2], length(g[[i]])))
		 df_updated = rbind(df_updated, df_tmp)
	}

}

# Read bin matrix:
W_bin = read.csv('W_bin.csv', row.names=1)

# Get the ID:
id = df[!duplicated(df[,c('name2', 'id_gbs')])][,c('name2', 'id_gbs')]

# Subset line names that were phenotyped and genotyped:
id = id[id$id_gbs %in% rownames(W_bin),]

# Subset adjusted means of the lines that were phenotyped and genotyped:
df_updated = df_updated[df_updated$name2 %in% id$name2,]

# Add column mapping the ID of the GBS:
df_updated= merge(df_updated, id, by='name2', sort=F) 

# Dropping levels:
df_updated$id_gbs = df_updated$id_gbs %>% droplevels

# Reodering the name of the data frame and eliminating the name2 identifier:
df_updated = df_updated[,c("id_gbs", "y_hat", "trait", "dap")] 
df_updated = df_updated[order(df_updated$trait),]
df_updated = df_updated[order(df_updated$dap),]


#---------------------------------------------Saving outputs-------------------------------------------------#

# Set directory:
setwd(paste0(OUT_PATH, '/processed_data'))

# Save adjusted means:
write.csv(df_updated, file=paste0("adjusted_means.csv"))
