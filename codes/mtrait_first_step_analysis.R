


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
df$id_gbs = df$id_gbs %>% as.factor()
df$block = df$block %>% as.factor()
df$loc = df$loc %>% as.factor()
df$year = df$year %>% as.factor()
df$trait = df$trait %>% as.factor()
df$dap = df$dap %>% as.integer()
df$drymass = df$drymass %>% as.integer()
df$height = df$drymass %>% as.integer()

