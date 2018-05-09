outs = fit['trn'].extract()

# Getting the predictions:
y_pred = dict()
y_pred['trn'] = outs['mu'].mean(axis=0) + X['trn'].dot(outs['beta'].mean(axis=0))
y_pred['dev'] = outs['mu'].mean(axis=0) + X['dev'].dot(outs['beta'].mean(axis=0))
y_pred['tst'] = outs['mu'].mean(axis=0) + X['tst'].dot(outs['beta'].mean(axis=0))

# Density plots of different data types:
sns.set_style('whitegrid')
ax = sns.kdeplot(y['trn'].flatten(), bw=0.5, label='train set', shade=True)
ax = sns.kdeplot(y['dev'].flatten(), bw=0.5, label='dev set', shade=True)
ax = sns.kdeplot(outs['y_gen'].mean(axis=0), bw=0.5, label='gen set', shade=True)
ax = sns.kdeplot(y_pred['dev'], bw=0.5, label='pred set', shade=True)
ax.set_title('Density of different data types')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.clf()

# Scatter plots of different data types:
tmp = dict()
tmp['trn'] = np.polyfit(y['trn'].flatten(), y_pred['trn'], 1)
tmp['dev'] = np.polyfit(y['dev'].flatten(), y_pred['dev'], 1)
tmp['tst'] = np.polyfit(y['tst'].flatten(), y_pred['tst'], 1)
plt.scatter(y['trn'].flatten(), y_pred['trn'], label="trn", alpha=0.3)
plt.plot(y['trn'].flatten(), tmp['trn'][0] * y['trn'].flatten() + tmp['trn'][1])
plt.scatter(y['dev'].flatten(), y_pred['dev'], label="dev", alpha=0.3)
plt.plot(y['dev'].flatten(), tmp['dev'][0] * y['dev'].flatten() + tmp['dev'][1])
plt.scatter(y['tst'].flatten(), y_pred['tst'], label="tst", alpha=0.3)
plt.plot(y['tst'].flatten(), tmp['tst'][0] * y['tst'].flatten() + tmp['tst'][1])
plt.legend()
plt.title('Scatter pattern of different data types')
plt.xlabel('Observed data')
plt.ylabel("Predicted data")
plt.xlim(0, 26)
plt.ylim(0, 26)
plt.show()
plt.clf()

# Printing rMSE:
rmse(y['trn'].flatten(), y_pred['trn'])
rmse(y['dev'].flatten(), y_pred['dev'])
rmse(y['tst'].flatten(), y_pred['tst'])

# Printing pearsonr:
pearsonr(y['trn'].flatten(), y_pred['trn'])[0]
pearsonr(y['dev'].flatten(), y_pred['dev'])[0]
pearsonr(y['tst'].flatten(), y_pred['tst'])[0]

# Printing r2:
r2_score(y['trn'].flatten(), y_pred['trn'])
r2_score(y['dev'].flatten(), y_pred['dev'])
r2_score(y['tst'].flatten(), y_pred['tst'])

