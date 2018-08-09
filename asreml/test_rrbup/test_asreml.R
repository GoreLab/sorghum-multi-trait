


# Prefix of the project:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"

# Load asreml:
setwd(paste0(prefix_proj, 'asreml'))
library(asreml)
asreml.lic(license = "asreml.lic", install = TRUE)

# Load the data:
setwd(paste0(prefix_proj, 'asreml/test_rrbup'))
hGP = read.table("hGP.txt", h = TRUE) 

# Subset markers:
# markers <- hGP[, 3:162] 
# markers <- as.matrix(markers) 
str(hGP)
 
# Fit rrBLUP model:
model1 <- asreml(predicted.value ~ 1,
	 			 random = ~ grp(markers), 
                 group = list(markers = 3:162), data = hGP) 

# Extract and plot marker effects: 
eff<-((model1)$coefficients$random[1:160]) #marker effects
plot((model1)$coefficients$random[1:160])



	