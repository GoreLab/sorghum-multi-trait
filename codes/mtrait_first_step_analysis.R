
#-------------------------------------------Loading libraries------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)

# Library for data analysis:
library(lme4)
library(sjstats)


#-----------------------------------------------Loading data-------------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

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

# Creating a list to receive the first stage results:
fit = list()
metrics = list()
g = list()


#------------------------------------------Biomass data analysis---------------------------------------------#

# Index for mapping the results:
index = "drymass"

# Subsetting part of the data set: 
df_tmp = df[df$trait == "DM"]

# Dropping levels not present in the data subset:
df_tmp$id_gbs = df_tmp$id_gbs %>% droplevels
df_tmp$block = df_tmp$block %>% droplevels
df_tmp$loc = df_tmp$loc %>% droplevels
df_tmp$year = df_tmp$year %>% droplevels
df_tmp$trait = df_tmp$trait %>% droplevels

# Fitting the model:
fit[[index]] = lmer(drymass ~ 1 + id_gbs + (1|block:loc) + (1|loc) + (1|year) + (1|id_gbs:loc) + (1|id_gbs:year), data=df_tmp)

# Getting the metrics to evaluate model performance:
metrics[["drymass-var"]] = re_var(fit[[index]]) %>% data.matrix
metrics[["drymass-cv"]] = cv(fit[[index]]) %>% data.matrix
metrics[["drymass-icc"]] = icc(fit[[index]]) %>% data.matrix

# Storing the corrected mean:
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
	df_tmp$loc = df_tmp$loc %>% droplevels
	df_tmp$year = df_tmp$year %>% droplevels
	df_tmp$trait = df_tmp$trait %>% droplevels

	# Fitting the model:
	fit[[index]] = lmer(height ~ 1 + id_gbs + (1|block:loc) + (1|loc) + (1|year) + (1|id_gbs:loc) + (1|id_gbs:year), data=df_tmp)

	# Getting the metrics to evaluate model performance:
	metrics[[paste0(index, "-var")]] = re_var(fit[[index]]) %>% data.matrix
	metrics[[paste0(index, "-cv")]] = cv(fit[[index]]) %>% data.matrix
	metrics[[paste0(index, "-icc")]] = icc(fit[[index]]) %>% data.matrix

	# Storing the corrected mean:
	g[[index]] = fixef(fit[[index]])[1] + fixef(fit[[index]])[-1]

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}


#---------------------------------------------Saving outputs-------------------------------------------------#

# Setting directory:
setwd(paste0(prefix_out, "outputs/first_step_analysis"))

# Saving results:
for (i in ls(metrics)) write.csv(metrics[i], file=paste0("metrics~", i, ".csv"))
for (i in ls(g)) write.csv(g[i], file=paste0("g~", i, ".csv"))

# Saving RData:
save.image("mtrait_first_step_analysis.RData")

