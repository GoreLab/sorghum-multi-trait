
#---------------------------------------Storing the best prediction------------------------------------------#

# Setting directory:
os.chdir(prefix_out + "outputs/rnaseq_imp/predicted_bins")


# Getting the index of the best predictions:
index = np.argsort(rmse_sets[:,1], axis=0)

# Getting the mean across dropout vectors:
Y_pred = np.mean(Y_pred_lst_sets[3], axis=2)

# Create empty submission dataframe
out = pd.DataFrame()

# Adding the predictions to the output file:
out[index_wbin[core].values[r]] = Y_pred[index[0], :]

# Insert ID and Predictions into dataframe
out.index = X['miss'].columns.values

# Output submission file
out.to_csv(index_wbin[core].values[r] + "_predicted.csv")

