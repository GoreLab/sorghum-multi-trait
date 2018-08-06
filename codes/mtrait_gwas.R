
#--------------------------------------------Load libraries--------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)

# Libraries for GWAS:
library(multtest)
library(gplots)
library(LDheatmap)
library(genetics)
library(ape)
library(EMMREML)
library(compiler) #this library is already installed in R
library("scatterplot3d")

# Loading GAPIT source codes:
source("http://zzlab.net/GAPIT/gapit_functions.txt")
source("http://zzlab.net/GAPIT/emma.txt")


#-----------------------------------------------Load data----------------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# Load adjusted means:
df = fread(paste0(prefix_out, "outputs/first_step_analysis/adjusted_means.csv"), header=TRUE)[,-1]

# Change classes:
df$id_gbs = df$id_gbs %>% as.factor %>% droplevels
df$trait = df$trait %>% as.factor %>% droplevels
df$dap = df$dap %>% as.factor %>% droplevels

# Structure of the data frame:
str(df)

# Read marker data:
M = fread(paste0(prefix_out, "data/M.csv"), header=TRUE)[,-1]

# Read loci information:
loci_info = fread(paste0(prefix_out, "data/gbs_info.csv"), header=TRUE)[,-1]

#---------------------------------------Prepare data format for GAPIT----------------------------------------#

# Creating the data frame:
myY = df %>% filter(dap != 'NA') %>%
 			 select(id_gbs, dap, y_hat) %>%
 			 spread(key = dap, value=y_hat)

# Reordering columns:
myY = myY[c('id_gbs', '30', '45', '60', '75', '90', '105', '120')]

# Changing column names:
colnames(myY) = c('ID', 'HT-30', 'HT-45', 'HT-60', 'HT-75', 'HT-90', 'HT-105', 'HT-120')

# Loading raw data:
df_raw = fread(paste0(prefix_out, "data/df.csv"), header=TRUE)[,-1]
df_raw$id_gbs = df_raw$id_gbs %>% as.factor %>% droplevels
df_raw$trait = df_raw$trait %>% as.factor %>% droplevels
df_raw$dap = df_raw$dap %>% as.factor %>% droplevels
df_raw$block = df_raw$block %>% as.factor %>% droplevels
df_raw$loc = df_raw$loc %>% as.factor %>% droplevels
df_raw$year = df_raw$year %>% as.factor %>% droplevels

str(df_raw)

