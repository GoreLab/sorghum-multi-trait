# Printing train pearsonr:
pearsonr(y['trn'].flatten(), y_pred['trn'])[0]

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev pearsonr:
pearsonr(y['dev'].flatten(), y_pred['dev'])[0]

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test pearsonr:
pearsonr(y['tst'].flatten(), y_pred['tst'])[0]

