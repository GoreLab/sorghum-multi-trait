
t='DM'

for (j in unique(df$env)) {

	df_tmp = df[df$env == j]
	for (i in unique(as.character(df_tmp$name2))) {

		if (t == 'DM') {
			mask = df_tmp$name2==i & df_tmp$trait=='DM' #& (!is.na(df$drymass))
		}
		if (str_detect(t, 'PH')) {
			mask = df_tmp$name2==i & df_tmp$trait=='PH' & df_tmp$dap == str_split(t, pattern="_", simplify = TRUE)[,2] #& (!is.na(df$height))
		}

		n_env_tmp = c(length(unique(as.character(df_tmp[mask, ]$env))))
		n_plot_tmp = c(nrow(df_tmp[mask, ]))
		names(n_env_tmp) = i
		names(n_plot_tmp) = i

		if (i==unique(as.character(df_tmp$name2))[1]) {

			n_env = n_env_tmp
			n_plot = n_plot_tmp

		}
		else {

			n_env = c(n_env, n_env_tmp)
			n_plot = c(n_plot, n_plot_tmp)

		}

	}

	print(paste0('All lines in  environment ', j, ' are:'))
	print(n_plot)

	print('------------------------------------------------------------')

	print(paste0('The check lines in environment ', j, ' are:'))
	print(n_plot[n_plot>4])

	print('------------------------------------------------------------')

}

[1] "The check lines in environment EF_16 are:"
Pacesetter   PI148084   PI148089   PI276801   NSL50748   PI524948 
        16         16         16         16         16         16 

[1] "The check lines in environment FF_16 are:"
Pacesetter   PI148084   PI148089   PI276801   NSL50748   PI524948 
        16         16         16         16         16         16

[1] "The check lines in environment EF_17 are:"
  PI524948        SPX   PI660560   PI525882   PI148089   PI276801 Pacesetter 
        16         17         16         16         16         16         16

[1] "The check lines in environment MW_17 are:"
Pacesetter   PI276801   PI525882   PI660560   PI148089   PI524948        SPX 
        16         16         16         16         16         16         17 


##########################


df_tmp = df[1][['plot', 'name1', 'name2', 'loc', 'set', 'block', 'drymass']]

df_tmp['drymass'] = df_tmp['drymass']*2.241699

# Pacesetter   PI276801   PI525882   PI660560   PI148089   PI524948        SPX 
df_tmp[(df_tmp['name2'] == 'SPX') & (df_tmp['loc']=='MW')]

# Raw data frame:
          plot    name1 name2 loc set  block    drymass
962   17MW0081  CHR-SPX   SPX  MW  Q2      1  34.782771
1056  17MW0005  CHR-SPX   SPX  MW  Q2      1  38.383935
1128  17MW0008  CHR-SPX   SPX  MW  Q2      1  35.605454
1136  17MW0328  CHR-SPX   SPX  MW  Q2      2  33.692903
1177  17MW0071  CHR-SPX   SPX  MW  Q2      1  36.303690
1180  17MW0170  CHR-SPX   SPX  MW  Q2      1  37.362899
1184  17MW0330  CHR-SPX   SPX  MW  Q2      2  38.508451
1186  17MW0410  CHR-SPX   SPX  MW  Q2      2        NaN
1201  17MW0070  CHR-SPX   SPX  MW  Q2      5  26.969886
1218  17MW0731  CHR-SPX   SPX  MW  Q1      8  38.333092
1285  17MW0547  CHR-SPX   SPX  MW  Q1      7  33.641885
1403  17MW0462  CHR-SPX   SPX  MW  Q2      6  34.853430
1407  17MW0622  CHR-SPX   SPX  MW  Q1      7        NaN
1587  17MW0134  CHR-SPX   SPX  MW  Q4      9  44.664440
1637  17MW0212  CHR-SPX   SPX  MW  Q4      9  44.006127
1892  17MW0839  CHR-SPX   SPX  MW  Q3     16  39.671729
1914  17MW0760  CHR-SPX   SPX  MW  Q3     16  24.997362

####################################3

data.frame(df[df$name2=='SPX' & df$loc=='MW' & df$trait=='DM'])

# Merged data frame:
   name2 id_gbs block loc year trait dap  drymass height   env
1    SPX            1  MW   17    DM  NA 34.78277     NA MW_17
2    SPX            1  MW   17    DM  NA 38.38394     NA MW_17
3    SPX            1  MW   17    DM  NA 35.60545     NA MW_17
4    SPX            2  MW   17    DM  NA 33.69290     NA MW_17
5    SPX            1  MW   17    DM  NA 36.30369     NA MW_17
6    SPX            1  MW   17    DM  NA 37.36290     NA MW_17
7    SPX            2  MW   17    DM  NA 38.50845     NA MW_17
8    SPX            2  MW   17    DM  NA       NA     NA MW_17
9    SPX            5  MW   17    DM  NA 26.96989     NA MW_17
10   SPX            8  MW   17    DM  NA 38.33309     NA MW_17
11   SPX            7  MW   17    DM  NA 33.64189     NA MW_17
12   SPX            6  MW   17    DM  NA 34.85343     NA MW_17
13   SPX            7  MW   17    DM  NA       NA     NA MW_17
14   SPX            9  MW   17    DM  NA 44.66444     NA MW_17
15   SPX            9  MW   17    DM  NA 44.00613     NA MW_17
16   SPX           16  MW   17    DM  NA 39.67173     NA MW_17
17   SPX           16  MW   17    DM  NA 24.99736     NA MW_17


############################

# PI524948        SPX   PI660560   PI525882   PI148089   PI276801 Pacesetter 
df_tmp[(df_tmp['name2'] == 'SPX') & (df_tmp['loc']=='EF')]

# Raw data frame:
         plot    name1 name2 loc set  block    drymass
9    17EF0400  CHR-SPX   SPX  EF  Q3      2  35.355284
71   17EF0958  CHR-SPX   SPX  EF  Q4      4  32.677980
148  17EF0167  CHR-SPX   SPX  EF  Q3      1  37.163391
239  17EF0951  CHR-SPX   SPX  EF  Q4      4  30.272203
550  17EF0903  CHR-SPX   SPX  EF  Q2     12  29.341817
595  17EF0776  CHR-SPX   SPX  EF  Q2     12  33.793199
615  17EF0615  CHR-SPX   SPX  EF  Q2     11  31.435930
630  17EF0267  CHR-SPX   SPX  EF  Q1     10  28.914074
665  17EF0693  CHR-SPX   SPX  EF  Q2     11  32.671035
672  17EF0029  CHR-SPX   SPX  EF  Q1      9  28.657560
692  17EF0829  CHR-SPX   SPX  EF  Q2     12  31.836778
718  17EF0910  CHR-SPX   SPX  EF  Q2     12  22.177053
778  17EF0433  CHR-SPX   SPX  EF  Q1     14  24.238532
832  17EF0675  CHR-SPX   SPX  EF  Q2     15  36.523117
927  17EF0602  CHR-SPX   SPX  EF  Q2     15  31.448681
928  17EF0679  CHR-SPX   SPX  EF  Q2     15  34.093394
933  17EF0842  CHR-SPX   SPX  EF  Q2     16  28.443063

####################################3

data.frame(df[df$name2=='SPX' & df$loc=='EF' & df$trait=='DM'])

# Merged data frame:
   name2 id_gbs block loc year trait dap  drymass height   env
1    SPX            2  EF   17    DM  NA 35.35528     NA EF_17
2    SPX            4  EF   17    DM  NA 32.67798     NA EF_17
3    SPX            1  EF   17    DM  NA 37.16339     NA EF_17
4    SPX            4  EF   17    DM  NA 30.27220     NA EF_17
5    SPX           12  EF   17    DM  NA 29.34182     NA EF_17
6    SPX           12  EF   17    DM  NA 33.79320     NA EF_17
7    SPX           11  EF   17    DM  NA 31.43593     NA EF_17
8    SPX           10  EF   17    DM  NA 28.91407     NA EF_17
9    SPX           11  EF   17    DM  NA 32.67104     NA EF_17
10   SPX            9  EF   17    DM  NA 28.65756     NA EF_17
11   SPX           12  EF   17    DM  NA 31.83678     NA EF_17
12   SPX           12  EF   17    DM  NA 22.17705     NA EF_17
13   SPX           14  EF   17    DM  NA 24.23853     NA EF_17
14   SPX           15  EF   17    DM  NA 36.52312     NA EF_17
15   SPX           15  EF   17    DM  NA 31.44868     NA EF_17
16   SPX           15  EF   17    DM  NA 34.09339     NA EF_17
17   SPX           16  EF   17    DM  NA 28.44306     NA EF_17


############################

length(unique(df$name2[!is.na(df$drymass)]))
length(unique(df$name2[!is.na(df$height[df$dap=='30'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='45'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='60'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='75'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='90'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='105'])]))
length(unique(df$name2[!is.na(df$height[df$dap=='120'])]))

# Total number of entries independent with there is missing data or not:
871

# Number of lines with phenotypes for drymass:
860

# Number of lines with phenotypes for plant height across all time points:
871

