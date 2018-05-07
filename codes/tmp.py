for (i in 1:length(tmp)) {

	X_tmp = data.matrix(fread(paste0('x_',tmp[i],'.csv'), header=TRUE))
	rownames(X_tmp) = X_tmp[,1]
	X[[tmp[i]]] = X_tmp[,-1]

}




