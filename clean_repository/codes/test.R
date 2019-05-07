
t='PH_30'

for (j in unique(df$env)) {

	df_tmp = df[df$env == j]
	for (i in unique(as.character(df_tmp$id_gbs))) {

		if (t == 'DM') {
			mask = df_tmp$id_gbs==i & df_tmp$trait=='DM' #& (!is.na(df$drymass))
		}
		if (str_detect(t, 'PH')) {
			mask = df_tmp$id_gbs==i & df_tmp$trait=='PH' & df_tmp$dap == str_split(t, pattern="_", simplify = TRUE)[,2] #& (!is.na(df$height))
		}

		n_env_tmp = c(length(unique(as.character(df_tmp[mask, ]$env))))
		n_plot_tmp = c(nrow(df_tmp[mask, ]))
		names(n_env_tmp) = i
		names(n_plot_tmp) = i

		if (i==unique(as.character(df_tmp$id_gbs))[1]) {

			n_env = n_env_tmp
			n_plot = n_plot_tmp

		}
		else {

			n_env = c(n_env, n_env_tmp)
			n_plot = c(n_plot, n_plot_tmp)

		}

	}
	print(paste0('The check lines in environment ', j, ' is:'))
	print(n_plot[n_plot>4])

}