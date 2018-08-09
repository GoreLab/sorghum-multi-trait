
rm(list=ls());ls()
setwd("/media/kaio/Arquivos Kaio/Tese-Embrapa_Dez_2016/rr-blup-asreml-R")

library(asreml)

hGP <- read.table("hGP.txt", h=T) 

markers <- hGP[, 3:162] 
markers <- as.matrix(markers) 
str(hGP)
 
####It worked. 
###
model1 <- asreml(predicted.value ~ 1, random = ~ grp(markers), 
                 group = list(markers = 3:162), data = hGP) 
 
summary(model1)$random

eff<-((model1)$coefficients$random[1:160]) #marker effects

plot((model1)$coefficients$random[1:160])


###################
####fazer o rr-blup 

rm(list=ls());ls()

pheno<-read.table("pheno.txt", sep="\t")
pheno<-as.data.frame(pheno[,2])
str(pheno)
pheno1<-as.numeric(pheno[,1])


geno<-read.table("geno.txt", sep=",")
geno<-as.matrix(geno) 
dim(geno)

##
library(rrBLUP)
###body(mixed.solve)
length(pheno)
model2=mixed.solve(pheno1,Z=geno)

model2$u #marker effects 
model2$beta #mean 
pred_pheno2<-  geno1 %*% model2$u # GEBV 


##########
#### cross validation 5 fold

cor<- matrix(NA, 375, 5)

for(j in 1:50)
{ 
  
  sample.size <- as.matrix(nrow(pheno)) 
  sequence.sample <- rep(1:sample.size) 
  random.sample <- sample(1:sample.size, replace = FALSE) 
  increment <- ceiling(sample.size/5) 
  for(i in 0:4){
    print(c(j,i))
    pred <- 
      as.matrix(random.sample[((increment*i)+1):min(((increment*i)+increment),sample.size)]) 
    train <- 
      as.matrix(random.sample[-(((increment*i)+1):min(((increment*i)+increment), sample.size))])
    geno.train <- as.matrix(geno[train,]) 
    rownames(geno.train)<- NULL 
    pheno.train <- as.matrix(pheno[train,1])
    rownames(pheno.train)<- NULL 
    geno.pred <- as.matrix(geno[pred,]) 
    rownames(geno.pred)<-NULL 
    pheno.pred <- as.matrix(pheno[pred,1])
    rownames(pheno.pred)<-NULL 
    
    ## Next lines used for rrBLUP  
        
    model2 <- mixed.solve(pheno.train,Z=geno.train)
    model2$u #marker effects 
    model2$beta #mean 
    pred_pheno2 <- geno.pred %*% model2$u # matrix multiplication to get vector of predicted phenotypes 
    pred_pheno <- pred_pheno2[,1]+model2$beta 
    pa2 <-(cor(pheno.pred,pred_pheno2)) # cor(EBVs,GEBVs) 
    
    cor[j,(i+1)] = rbind (pa2)
    #values2[(i+1),j] = rbind(pa2) 
    print(pa2)
    print( "---")
    
  } 
}


