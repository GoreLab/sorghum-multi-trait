

library(data.table)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(Matrix)
library(qdapTools)
library(rrBLUP)
library(EMMREML)

l=20
n<-15
m<-40

M<-matrix(rbinom(m*l,2,.2),nrow=l)
rownames(M)<-paste("l",1:nrow(M))
beta1<-rnorm(m)*exp(rbinom(m,5,.2))
beta2<-rnorm(m)*exp(rbinom(m,5,.1))
beta3<- rnorm(m)*exp(rbinom(m,5,.1))+beta2

g1<-M%*%beta1
g2<-M%*%beta2
g3<-M%*%beta3
e1<-sd(g1)*rnorm(l)
e2<-(-e1*2*sd(g2)/sd(g1)+.25*sd(g2)/sd(g1)*rnorm(l))
e3<-1*(e1*.25*sd(g2)/sd(g1)+.25*sd(g2)/sd(g1)*rnorm(l))

y1<-10+g1+e1
y2<--50+g2+e2
y3<--5+g3+e3

Y<-rbind(t(y1),t(y2), t(y3))
colnames(Y)<-rownames(M)
cov(t(Y))
Y[1:3,1:5]

K<-cov(t(M))
K<-K/mean(diag(K))
rownames(K)<-colnames(K)<-rownames(M)
X<-matrix(1,nrow=1,ncol=l)
colnames(X)<-rownames(M)
Z<-diag(l)
rownames(Z)<-colnames(Z)<-rownames(M)
SampleTrain<-sample(rownames(Z),n)
Ztrain<-Z[rownames(Z)%in%SampleTrain,]
Ztest<-Z[!(rownames(Z)%in%SampleTrain),]

# For a quick answer, tolpar is set to 1e-4. Correct this in practice.
outfunc <- emmremlMultivariate(Y = Y%*%t(Ztrain),
			  				   X = X%*%t(Ztrain),
							   Z = t(Ztrain),
							   K = K,
							   tolpar = 1e-4,
							   varBhat = FALSE, varGhat = FALSE, PEVGhat = FALSE, test = FALSE)

Yhattest<-outfunc$Gpred%*%t(Ztest)
cor(cbind(Ztest%*%Y[1,],Ztest%*%outfunc$Gpred[1,],
Ztest%*%Y[2,],Ztest%*%outfunc$Gpred[2,],Ztest%*%Y[3,],Ztest%*%outfunc$Gpred[3,]))

outfuncRidgeReg <- emmremlMultivariate(Y = Y%*%t(Ztrain),
									   X = X%*%t(Ztrain), 
									   Z = t(Ztrain%*%M),
									   K = diag(m),
									   tolpar = 1e-5,
									   varBhat = FALSE, varGhat = FALSE, PEVGhat = FALSE, test = FALSE)


Gpred2<-outfuncRidgeReg$Gpred%*%t(M)
cor(Ztest%*%Y[1,],Ztest%*%Gpred2[1,])
cor(Ztest%*%Y[2,],Ztest%*%Gpred2[2,])
cor(Ztest%*%Y[3,],Ztest%*%Gpred2[3,])

##############

cat y_cv1_bn_trn_files.txt
y_cv1_drymass_k0_trn.csv
y_cv1_drymass_k1_trn.csv
y_cv1_drymass_k2_trn.csv
y_cv1_drymass_k3_trn.csv
y_cv1_drymass_k4_trn.csv
y_cv1_height_k0_trn.csv
y_cv1_height_k1_trn.csv
y_cv1_height_k2_trn.csv
y_cv1_height_k3_trn.csv
y_cv1_height_k4_trn.csv


cat y_cv1_pbn_trn_files.txt
y_cv1_drymass_k0_trn.csv&y_cv1_height_k0_trn.csv
y_cv1_drymass_k1_trn.csv&y_cv1_height_k1_trn.csv
y_cv1_drymass_k2_trn.csv&y_cv1_height_k2_trn.csv
y_cv1_drymass_k3_trn.csv&y_cv1_height_k3_trn.csv
y_cv1_drymass_k4_trn.csv&y_cv1_height_k4_trn.csv


cat y_cv1_dbn_trn_files.txt
y_cv1_height_k0_trn.csv
y_cv1_height_k1_trn.csv
y_cv1_height_k2_trn.csv
y_cv1_height_k3_trn.csv
y_cv1_height_k4_trn.csv


cat y_cv2_bn_trn_files.txt
y_cv2-105~only_height_trn.csv
y_cv2-30~only_height_trn.csv
y_cv2-45~only_height_trn.csv
y_cv2-60~only_height_trn.csv
y_cv2-75~only_height_trn.csv
y_cv2-90~only_height_trn.csv


cat y_cv2_pbn_trn_files.txt
y_cv2_drymass_trn.csv&y_cv2-105~only_height_trn.csv
y_cv2_drymass_trn.csv&y_cv2-30~only_height_trn.csv
y_cv2_drymass_trn.csv&y_cv2-45~only_height_trn.csv
y_cv2_drymass_trn.csv&y_cv2-60~only_height_trn.csv
y_cv2_drymass_trn.csv&y_cv2-75~only_height_trn.csv
y_cv2_drymass_trn.csv&y_cv2-90~only_height_trn.csv


cat y_cv2_dbn_trn_files.txt
y_cv2-30~105_height_trn.csv
y_cv2-30~45_height_trn.csv
y_cv2-30~60_height_trn.csv
y_cv2-30~75_height_trn.csv
y_cv2-30~90_height_trn.csv




