
#-------------------------------------------Loading libraries------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)

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

# Fitting the model:
fit[[index]] = lmer(drymass ~ 1 + (1|id_gbs) + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

# Receive variance components:
v_ge = summary(fit[[index]])$varcor['id_gbs:env']['id_gbs:env'] %>% unlist
v_g = summary(fit[[index]])$varcor['id_gbs']['id_gbs'] %>% unlist
v_be = summary(fit[[index]])$varcor['block:env']['block:env'] %>% unlist
v_env = summary(fit[[index]])$varcor['env']['env'] %>% unlist
v_e = (summary(fit[[index]])$sigma)^2 %>% unlist

# Compute herdability:
h = v_g / (v_g + v_ge/nlevels(df_tmp$loc) + v_be + v_env + v_e/(nlevels(df_tmp$loc)*nlevels(df_tmp$block)))

# Getting the metrics to evaluate model performance:
metrics[["drymass-h2"]] = h


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
	fit[[index]] = lmer(height ~ 1 + (1|id_gbs) + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

	# Receive variance components:
	v_ge = summary(fit[[index]])$varcor['id_gbs:env']['id_gbs:env'] %>% unlist
	v_g = summary(fit[[index]])$varcor['id_gbs']['id_gbs'] %>% unlist
	v_be = summary(fit[[index]])$varcor['block:env']['block:env'] %>% unlist
	v_env = summary(fit[[index]])$varcor['env']['env'] %>% unlist
	v_e = (summary(fit[[index]])$sigma)^2 %>% unlist

	# Compute herdability:
	h = v_g / (v_g + v_ge/nlevels(df_tmp$loc) + v_be + v_env + v_e/(nlevels(df_tmp$loc)*nlevels(df_tmp$block)))

	# Getting the metrics to evaluate model performance:
	metrics[[paste0(index, "-h2")]] = h

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

#---------------------------------------------Saving outputs-------------------------------------------------#

# Transforming list into table:
h_table = qdapTools::list_vect2df(metrics, col1 = "dap", col2 = "structure", col3 = "h2")

# Setting directory:
setwd(paste0(prefix_out, "outputs/first_step_analysis"))

# Saving results:
write.csv(h_table, file="mtrait_first_step_analysis_herdability_table.txt")

# Saving RData:
save.image("mtrait_first_step_analysis_herdability.RData")

# # Loading data:
# setwd(paste0(prefix_out, "outputs/first_step_analysis"))
# load("mtrait_first_step_analysis.RData")
