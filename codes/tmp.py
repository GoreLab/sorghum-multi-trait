
# Density plots of different data types:
sns.set_style('whitegrid')
ax = sns.kdeplot(y['trn'].values.flatten(), bw=0.5, label='train set', shade=True)
ax = sns.kdeplot(y['dev'].values.flatten(), bw=0.5, label='dev set', shade=True)
ax = sns.kdeplot(outs['y_gen'].mean(axis=0), bw=0.5, label='gen set', shade=True)
ax = sns.kdeplot(y_pred['dev'], bw=0.5, label='pred set', shade=True)
ax.set_title('Density of different data types')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.clf()

# Scatter plots of different data types:
tmp = dict()
tmp['trn'] = np.polyfit(y['trn'].values.flatten(), y_pred['trn'], 1)
tmp['dev'] = np.polyfit(y['dev'].values.flatten(), y_pred['dev'], 1)
tmp['tst'] = np.polyfit(y['tst'].values.flatten(), y_pred['tst'], 1)
plt.scatter(y['trn'].values.flatten(), y_pred['trn'], label="trn", alpha=0.3)
plt.plot(y['trn'].values.flatten(), tmp['trn'][0] * y['trn'].values.flatten() + tmp['trn'][1])
plt.scatter(y['dev'].values.flatten(), y_pred['dev'], label="dev", alpha=0.3)
plt.plot(y['dev'].values.flatten(), tmp['dev'][0] * y['dev'].values.flatten() + tmp['dev'][1])
plt.scatter(y['tst'].values.flatten(), y_pred['tst'], label="tst", alpha=0.3)
plt.plot(y['tst'].values.flatten(), tmp['tst'][0] * y['tst'].values.flatten() + tmp['tst'][1])
plt.legend()
plt.title('Scatter pattern of different data types')
plt.xlabel('Observed data')
plt.ylabel("Predicted data")
plt.xlim(0, 26)
plt.ylim(0, 26)
plt.show()
plt.clf()

# Printing rMSE:
rmse(y['trn'].values.flatten(), y_pred['trn'])
rmse(y['dev'].values.flatten(), y_pred['dev'])
rmse(y['tst'].values.flatten(), y_pred['tst'])

# Printing pearsonr:
pearsonr(y['trn'].values.flatten(), y_pred['trn'])[0]
pearsonr(y['dev'].values.flatten(), y_pred['dev'])[0]
pearsonr(y['tst'].values.flatten(), y_pred['tst'])[0]

# Printing r2:
r2_score(y['trn'].values.flatten(), y_pred['trn'])
r2_score(y['dev'].values.flatten(), y_pred['dev'])
r2_score(y['tst'].values.flatten(), y_pred['tst'])


