








namevec <- rownames(summary(fit[[index]])$varcomp)
names(asrMod[[1]]) <- names(asrMod[[2]]) <- namevec 

nadiv:::pin(asrMod, h2 ~ V3 / (V1 + V2 + V3 + V4 + V5))


summary(fit[[index]])$varcomp


