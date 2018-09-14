
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
g = list()

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
fit[[index]] = lmer(drymass ~ 1 + id_gbs + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

# Getting the metrics to evaluate model performance:
metrics[["drymass-cv"]] = cv(fit[[index]]) %>% data.matrix

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
	df_tmp$env = df_tmp$env %>% droplevels

	# Fitting the model:
	fit[[index]] = lmer(height ~ 1 + id_gbs + (1|block:env) + (1|env) + (1|id_gbs:env), data=df_tmp)

	# Getting the metrics to evaluate model performance:
	metrics[[paste0(index, "-cv")]] = cv(fit[[index]]) %>% data.matrix

	# Storing the corrected mean:
	g[[index]] = fixef(fit[[index]])[1] + fixef(fit[[index]])[-1]

	# Printing current analysis:
	print(paste0("Analysis ", index, " done!"))

}

# Transform list to data frame:
g = qdapTools::list_vect2df(g, col1 = "trait", col2 = "id_gbs", col3 = "y_hat")

# Split the trait columns into trait and DAP:
trait_groups = g$trait %>% str_split(., pattern="_", simplify = TRUE)

# Adding the new columns of trait and DAP to the data frame:
g = g %>% select(-trait) %>% cbind(.,trait_groups)
colnames(g) = c('id_gbs', 'y_hat', 'trait', 'dap')

# Adding NA to the missing entries of dry mass:
g$dap = g$dap %>% as.character()
g$dap[g$dap == ""] = "NA"
g$dap = g$dap %>% as.factor()
	
# Getting just the name of the lines:
id_gbs_groups = g$id_gbs %>% as.character %>% str_split(., pattern="gbs", simplify=TRUE)

# Adding to the data frame only the name of the lines and changing classes:
g$id_gbs = id_gbs_groups[,2] %>% as.factor
g$trait = g$trait %>% as.factor
g$dap = g$dap %>% as.factor
g = g[,c("id_gbs", "trait", "dap", "y_hat")]


#---------------------------------------------Saving outputs-------------------------------------------------#

# Setting directory:
setwd(paste0(prefix_out, "outputs/first_step_analysis"))

# Saving results:
for (i in ls(metrics)) write.csv(metrics[i], file=paste0("metrics~", i, ".csv"))
write.csv(g, file=paste0("adjusted_means.csv"))

# Saving RData:
save.image("mtrait_first_step_analysis.RData")

# # Loading data:
# setwd(paste0(prefix_out, "outputs/first_step_analysis"))
# load("mtrait_first_step_analysis.RData")



# c = rep(49.9, 3)
# a = rep(134, 3)
# r = rep(1.96, 3)

# out = matrix(NA, 3, 10)

# for (t in 0:10) {

# 	out[,t] = c / (1 + (a * exp(-r * t)))

# }

# plot(x=1:10, y=out[1,],type="line")
