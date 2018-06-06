
# Prefix of the directory of the project is in:
# prefix_proj = "/home/jhonathan/Documentos/mtrait-proj/"
prefix_proj = "/data1/aafgarci/jhonathan/mtrait-proj/"

# Prefix where the outputs will be saved:
# prefix_out = "/home/jhonathan/Documentos/resul_mtrait-proj/"
prefix_out = "/data1/aafgarci/jhonathan/resul_mtrait-proj/"


# Setting directory:
setwd(paste0(prefix_out, "data"))

# Loading marker data:
load("sorghum_PstI_info.RData")
load("sorghum_PstI_snps012.RData")
load("sorghum_PstI_taxa.RData")

write.csv(snps012, file="gbs.txt")
write.csv(taxa, file="taxa.txt")
write.csv(snp_info, file="gbs_info.txt")




