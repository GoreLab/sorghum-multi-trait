


# Set directory:
os.chdir(REPO_PATH + "/clean_repository/tables")

# Initialize a dictionary to receive the accuracies:
table = dict()

# Load correlation matrices:
table['MTi-GBLUP_fcv'] = np.round(pd.read_csv('acc_MTi-GBLUP_fcv.csv', header = 0, index_col=0).values,4)
table['MTr-GBLUP_fcv'] = np.round(pd.read_csv('acc_MTr-GBLUP_fcv.csv', header = 0, index_col=0).values,4)

# List of models to use:
model_set = ['bn', 'pbn', 'dbn']

# Generate accuracy heatmaps:
for i in model_set:
  # Heat map of the adjusted means across traits:
  table [i + '_fcv']= np.flip(np.flip(cor_dict['fcv_' + i],axis=1), axis=0)


table['bn_fcv']
table['pbn_fcv']
table['dbn_fcv']
table['MTi-GBLUP_fcv']
table['MTr-GBLUP_fcv']


(table['pbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100
np.nanmin((table['pbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)
np.nanmax((table['pbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)

(table['dbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100
np.nanmin((table['dbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)
np.nanmax((table['dbn_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)

(table['MTi-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100
np.nanmin((table['MTi-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)
np.nanmax((table['MTi-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)

(table['MTr-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100
np.nanmin((table['MTr-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)
np.nanmax((table['MTr-GBLUP_fcv'] - table['bn_fcv'])/table['bn_fcv']*100)


lines = pd.DataFrame(['Pacesetter', 'PI276801', 'PI148089', 'PI524948', 'NSL50748', 'PI148084', 'NSL50748', 'PI148084', 'PI525882', 'PI660560', 'SPX'],columns=['id'])

for i in lines.id:
  print('line ', i, sum(df.id_gbs.unique() == i))



tmp1 = pd.DataFrame(['Pacesetter', 'PI276801', 'PI148089', 'PI524948', 'NSL50748', 'PI148084'], columns=['id'])
tmp2 = pd.DataFrame(['Pacesetter', 'PI276801', 'PI148089', 'PI524948', 'PI525882', 'PI660560', 'SPX'], columns=['id'])







