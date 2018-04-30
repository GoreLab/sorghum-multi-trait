

#---------------------------------------Loading libraries---------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(rlist)
library(Matrix)

# Library for plots:
library(ggplot2)
library(ggpubr)

# Prefix of the directory of the project is in (choose the directory to the desired machine by removing comment):
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Prefix where the outputs will be saved:
prefix_out = "/workdir/jp2476/repo/resul_mtrait-proj/"

#------------------------------------------Loading data-----------------------------------------#

# Setting directory:
setwd(paste0(prefix_out, "data"))

# Reading data:
df = data.table(read.csv('df.csv'))

# Changing dap to the factor class:
df$dap <- as.factor(df$dap)

# Changing class:
df = df %>% as.data.table()

# Height against days after planting, locations, and years:
p1 = df[(df$trait=='height') & (!is.na(df$height)),] %>% ggplot(aes(x=dap, y=height, group=dap)) + 
  	   			 		    	  geom_boxplot(aes(fill=dap))+ facet_grid(year ~ loc)


# Biomass against days after planting, locations, and years:
p2 = df[(df$trait=='biomass') & (!is.na(df$drymass)),] %>% ggplot(aes(x=loc, y=drymass, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# ADF against days after planting, locations, and years:
p3 = df[(df$trait=='biomass') & (!is.na(df$adf)),] %>% ggplot(aes(x=loc, y=adf, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# Moisture against days after planting, locations, and years:
p4 = df[(df$trait=='biomass') & (!is.na(df$moisture)),] %>% ggplot(aes(x=loc, y=moisture, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# NDF against days after planting, locations, and years:
p5 = df[(df$trait=='biomass') & (!is.na(df$ndf)),] %>% ggplot(aes(x=loc, y=ndf, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# Protein against days after planting, locations, and years:
p6 = df[(df$trait=='biomass') & (!is.na(df$protein)),] %>% ggplot(aes(x=loc, y=protein, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# Starch against days after planting, locations, and years:
p7 = df[(df$trait=='biomass') & (!is.na(df$starch)),] %>% ggplot(aes(x=loc, y=starch, group=loc)) + 
  	   			 		    	  geom_boxplot(aes(fill=loc))+ facet_grid(. ~ year)

# Height against days after planting, locations, and years:
p8 = df[(df$trait=='height') & (!is.na(df$height)),] %>% ggplot(aes(sample=height, color=dap)) +
 														 stat_qq() + 
 														 facet_grid(loc ~ year) + 
 														 ggtitle('height')

# Biomass against days after planting, locations, and years:
p9 = df[(df$trait=='biomass') & (!is.na(df$drymass)),] %>% ggplot(aes(sample=drymass, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('drymass')

# ADF against days after planting, locations, and years:
p10 = df[(df$trait=='biomass') & (!is.na(df$adf)),] %>% ggplot(aes(sample=adf, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('adf')

# Moisture against days after planting, locations, and years:
p11 = df[(df$trait=='biomass') & (!is.na(df$moisture)),] %>% ggplot(aes(sample=moisture, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('moisture')

# NDF against days after planting, locations, and years:
p12 = df[(df$trait=='biomass') & (!is.na(df$ndf)),] %>% ggplot(aes(sample=ndf, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('ndf')

# Protein against days after planting, locations, and years:
p13 = df[(df$trait=='biomass') & (!is.na(df$protein)),] %>% ggplot(aes(sample=protein, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('protein')

# Starch against days after planting, locations, and years:
p14 = df[(df$trait=='biomass') & (!is.na(df$starch)),] %>% ggplot(aes(sample=starch, color=loc)) +
 														  stat_qq() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('starch')


