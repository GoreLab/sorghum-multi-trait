
#--------------------------------------------Load libraries--------------------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)


#-------------------------------------Subset the information from flags--------------------------------------#

# Subset the arguments:
args=commandArgs(trailingOnly = TRUE)

print(args)


#-----------------------------------------------Load data----------------------------------------------------#

# # Prefix of the directory of the project is in:
# prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# # Prefix where the outputs will be saved:
# prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

# # Load adjusted means:
# df = fread(paste0(prefix_out, "outputs/first_step_analysis/adjusted_means.csv"), header=TRUE)[,-1]

# # Change classes:
# df$id_gbs = df$id_gbs %>% as.factor %>% droplevels
# df$trait = df$trait %>% as.factor %>% droplevels
# df$dap = df$dap %>% as.factor %>% droplevels

# # Structure of the data frame:
# str(df)

# # Read binned marker data:
# W_bin = fread(paste0(prefix_out, "data/W_bin.csv"), header=TRUE)[,-1]


# #--------------------------------------Prepare data format for sommer----------------------------------------#

# # Creating the data frame:
# df_sommer = df %>% filter((trait == 'height')) %>%
#  			 	   select(id_gbs, dap, y_hat) %>%
#  			 	   spread(key = dap, value=y_hat)

# # Reordering columns:
# df_sommer = df_sommer[c('id_gbs', '30', '45', '60', '75', '90', '105', '120')]

# # Changing column names:
# colnames(df_sommer) = c('ID', 'HT-30', 'HT-45', 'HT-60', 'HT-75', 'HT-90', 'HT-105', 'HT-120')



