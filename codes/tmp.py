
# Initialize a variable to receive metrics:
rmse_sets = np.empty([n_alt, n_core, n_sets])
cor_sets = np.empty([n_alt, n_core, n_sets])
r2_sets = np.empty([n_alt, n_core, n_sets])

# Getting metrics:
for c in range(n_sets):
  # Getting final predictions:
  Y_pred = np.mean(Y_pred_lst_sets[c], axis=2)
  # Computing RMSE of the prediction using the dev set:
  for m in [0,1,3,5]:
    # Getting data set:
    if c==0:
        y_tmp = y['trn'].values.flatten()
    elif c==1:
        y_tmp = y['dev'].values.flatten()
    else:
        y_tmp = y['tst'].values.flatten()
    for alt in range(n_alt):
        rmse_sets[alt, m, c] = rmse(y_tmp, Y_pred[alt,:,m])
        cor_sets[alt, m, c] = pearsonr(y_tmp, Y_pred[alt,:,m])[0]
        r2_sets[alt, m, c] = r2_score(y_tmp, Y_pred[alt,:,m])

# Printing RMSE:
print(np.round(np.sort(rmse_sets[:,:,0], axis=0),4))
print(np.round(np.sort(rmse_sets[:,:,1], axis=0),4))
print(np.round(np.sort(rmse_sets[:,:,2], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,:,0], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,:,1], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,:,2], axis=0),4))

# Printing Pearson correlation:
print(np.round(np.sort(cor_sets[:,:,0], axis=0)[::-1],4))
print(np.round(np.sort(cor_sets[:,:,1], axis=0)[::-1],4))
print(np.round(np.sort(cor_sets[:,:,2], axis=0)[::-1],4))
print(np.round(np.argsort(cor_sets[:,:,0], axis=0)[::-1],4))
print(np.round(np.argsort(cor_sets[:,:,1], axis=0)[::-1],4))
print(np.round(np.argsort(cor_sets[:,:,2], axis=0)[::-1],4))

# Printing Mutual Information Criteria:
print(np.round(np.sort(mic_sets[:,:,0], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,:,1], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,:,2], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,:,0], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,:,1], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,:,2], axis=0)[::-1],4))


# Printing just one alternative:
m = 2; alt = 12;
print(np.round(rmse_sets[alt, m , 0],4))
print(np.round(rmse_sets[alt, m , 1],4))
print(np.round(rmse_sets[alt, m ,2],4))

for m in range(n_core):
# Getting data set:
if c==0:
    y_tmp = y['trn'].values.flatten()
elif c==1:
    y_tmp = y['dev'].values.flatten()
else:
    y_tmp = y['tst'].values.flatten()
for alt in range(n_alt):
    rmse_sets[alt, m, c] = rmse(y_tmp, Y_pred[alt,:,m])
    cor_sets[alt, m, c] = pearsonr(y_tmp, Y_pred[alt,:,m])[0]
    r2_sets[alt, m, c] = r2_score(y_tmp, Y_pred[alt,:,m])
