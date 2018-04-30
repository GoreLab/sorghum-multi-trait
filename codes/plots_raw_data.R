

#---------------------------------------Loading libraries---------------------------------------#

# Libraries to manipulate data:
library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)

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

# Setting directory to store plots:
setwd(paste0(prefix_out, "plots"))

#--------------------------------------------Box plots------------------------------------------#

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

# Saving plots:
ggsave(filename="height_boxplot.pdf", plot=p1, dpi=150, device="pdf")
ggsave(filename="drymass_boxplot.pdf", plot=p2, dpi=150, device="pdf")
ggsave(filename="adf_boxplot.pdf", plot=p3, dpi=150, device="pdf")
ggsave(filename="moisture_boxplot.pdf", plot=p4, dpi=150, device="pdf")
ggsave(filename="ndf_boxplot.pdf", plot=p5, dpi=150, device="pdf")
ggsave(filename="protein_boxplot.pdf", plot=p6, dpi=150, device="pdf")
ggsave(filename="starch_boxplot.pdf", plot=p7, dpi=150, device="pdf")


#---------------------------------------------qq plots------------------------------------------#

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

# Saving plots:
ggsave(filename="height_qqplot.pdf", plot=p8, dpi=150, device="pdf")
ggsave(filename="drymass_qqplot.pdf", plot=p9, dpi=150, device="pdf")
ggsave(filename="adf_qqplot.pdf", plot=p10, dpi=150, device="pdf")
ggsave(filename="moisture_qqplot.pdf", plot=p11, dpi=150, device="pdf")
ggsave(filename="ndf_qqplot.pdf", plot=p12, dpi=150, device="pdf")
ggsave(filename="protein_qqplot.pdf", plot=p13, dpi=150, device="pdf")
ggsave(filename="starch_qqplot.pdf", plot=p14, dpi=150, device="pdf")


#-----------------------------------------Histograms plots--------------------------------------#


# Height against days after planting, locations, and years:
p15 = df[(df$trait=='height') & (!is.na(df$height)),] %>% ggplot(aes(x=height, color=dap)) +
 														 geom_histogram() + 
 														 facet_grid(loc ~ year) + 
 														 ggtitle('height')

# Biomass against days after planting, locations, and years:
p16 = df[(df$trait=='biomass') & (!is.na(df$drymass)),] %>% ggplot(aes(x=drymass, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('drymass')

# ADF against days after planting, locations, and years:
p17 = df[(df$trait=='biomass') & (!is.na(df$adf)),] %>% ggplot(aes(x=adf, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('adf')

# Moisture against days after planting, locations, and years:
p18 = df[(df$trait=='biomass') & (!is.na(df$moisture)),] %>% ggplot(aes(x=moisture, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('moisture')

# NDF against days after planting, locations, and years:
p19 = df[(df$trait=='biomass') & (!is.na(df$ndf)),] %>% ggplot(aes(x=ndf, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('ndf')

# Protein against days after planting, locations, and years:
p20 = df[(df$trait=='biomass') & (!is.na(df$protein)),] %>% ggplot(aes(x=protein, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('protein')

# Starch against days after planting, locations, and years:
p21 = df[(df$trait=='biomass') & (!is.na(df$starch)),] %>% ggplot(aes(x=starch, color=loc)) +
 														  geom_histogram() + 
 														  facet_grid(. ~ year) + 
 														  ggtitle('starch')


# Saving plots:
ggsave(filename="height_histogram.pdf", plot=p15, dpi=150, device="pdf")
ggsave(filename="drymass_histogram.pdf", plot=p16, dpi=150, device="pdf")
ggsave(filename="adf_histogram.pdf", plot=p17, dpi=150, device="pdf")
ggsave(filename="moisture_histogram.pdf", plot=p18, dpi=150, device="pdf")
ggsave(filename="ndf_histogram.pdf", plot=p19, dpi=150, device="pdf")
ggsave(filename="protein_histogram.pdf", plot=p20, dpi=150, device="pdf")
ggsave(filename="starch_histogram.pdf", plot=p21, dpi=150, device="pdf")

