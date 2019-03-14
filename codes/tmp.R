n_env = df_tmp$env %>% nlevels
n_rep = df_tmp$block %>% nlevels

# Re-create the output from a basic, univariate model in asreml-R: 
asrMod <- list(gammas = asreml_obj$gammas, gammas.type = asreml_obj$gammas.type, ai = asreml_obj$ai)

# Name objects:
names(asrMod[[1]]) <- names(asrMod[[2]]) <- names(asreml_obj$gammas)

# Compute the heritability and its standard deviation:
#---Observation: V4/3 is the var_GxE /#locations and V5/48 is the var_E /# locations * # blocks
formula = eval(parse(text=paste0('h2 ~ V3 / (V3 + V4/', n_env, '+ V5/', n_env*n_rep, ')')))
return(nadiv:::pin(asrMod, formula))
