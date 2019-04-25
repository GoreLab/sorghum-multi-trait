
#-------------------------------------------Loading libraries------------------------------------------------#

# Import python libraries:
library(optparse)
library(data.table)
library(magrittr)
library(lme4)
library(sjstats)

#-----------------------------------------Adding flags to the code-------------------------------------------#

# Set the flags:
option_list = list(
  make_option(c("-r", "--rpath"), type="character", default=NULL, 
              help="The path of the repository", metavar="character"),
  make_option(c("-o", "--opath"), type="character", default=NULL, 
              help="The path of the folder to receive outputs (also root of the processed data)", metavar="character")
) 

# Parse the arguments:
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

# Subset arguments:
# REPO_PATH = args$rpath
# OUT_PATH = args$opath
REPO_PATH = '/home/jhonathan/Documents/sorghum-multi-trait'
OUT_PATH = '/home/jhonathan/Documents/output_sorghum-multi-trait'

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
# df[df$env == 'FF_16',]$block %>% as.character() %>% unique %>% length
# df[df$env == 'EF_16',]$block %>% as.character() %>% unique %>% length
# df[df$env == 'EF_17',]$block %>% as.character() %>% unique %>% length
# df[df$env == 'MW_17',]$block %>% as.character() %>% unique %>% length
# df %>% str


#------------------------------------------Biomass data analysis---------------------------------------------#

# Initialize a list to receive the first stage results:
fit = list()
metrics = list()
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

# Get the metrics to evaluate model performance:
metrics[["drymass-cv"]] = cv(fit[[index]]) %>% data.matrix

# Store the corrected mean:
g[[index]] = fixef(fit[[index]])[1] + fixef(fit[[index]])[-1]



