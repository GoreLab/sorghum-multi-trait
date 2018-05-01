


# To reset the previous graph:
tf.reset_default_graph()

# Number of simulations:
n_sim=100

# Small epsilon value for batch norm
epsilon = 1e-7

# Number of data sets:
n_sets = 4

# Initializing lists:
Y_pred_lst_sets = []
Y_pred_lst = np.empty(shape=[n_alt, X['trn'].shape[1] , n_sim])

# Foward pass:
for c in range(n_sets):
	# For training set:
	if c==0:
	  X_tmp = X['trn']
	# For development set:
	if c==1:
	  X_tmp = X['dev']
	# For testing set:
	if c==2:
	  X_tmp = X['tst']
	# For missing set:
	if c==3:
	  X_tmp = X['miss']
	# Initializing list:
	Y_pred_lst = np.empty(shape=[n_alt, X_tmp.shape[1] , n_sim])
	for alt in range(n_alt):
	  # Importing results:
	  tf.reset_default_graph()
	  n_layers = n_layers_lst[alt]
	  session = tf.Session() 
	  save_path = prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + "/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + ".meta"
	  saver = tf.train.import_meta_graph(save_path, clear_devices=True)
	  save_path = prefix_out + "outputs/rnaseq_imp/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r)
	  saver.restore(session,tf.train.latest_checkpoint(save_path))
	  for sim in range(n_sim):
	    for l in range(n_layers):
	      # Initializing variables:
	      if l==0:
	        Z = []
	        A = []          
	      # Initializing batch norm variables:
	      if batch_mode==True and l==0:
	        batch_mean = []
	        batch_var = []
	        Z_norm = []
	        Z_tilde = []
	      # Initializing dropout variables:n_sets
	      if dropout_mode==True and l==0:
	        D = []
	      # Linear activation:
	      if l==0:
	        Z.append(np.dot(session.run("W1:0"), X_tmp) + session.run("B1:0"))            
	      if l!=0:
	        Z.append(np.dot(session.run("W{}:0".format(l+1)), A[l-1]) + session.run("B{}:0".format(l+1)))   
	      # Batch norm:
	      if batch_mode==True:
	        batch_mean.append(moment(Z[l], moment=1, axis=1))                                                         # Getting the mean across examples
	        batch_var.append(moment(Z[l], moment=2, axis=1))                                                          # Getting the variance across examples
	        batch_mean[l] = batch_mean[l].reshape([Z[l].shape[0], 1])                                                 # Reshaping moments
	        batch_var[l] = batch_var[l].reshape([Z[l].shape[0],1])                                                    # Reshaping moments    
	        Z_norm.append((Z[l] - batch_mean[l]) / np.sqrt(batch_var[l] + epsilon))                                     # Normalizing output of the linear combination
	        Z_tilde.append(session.run("gamma{}:0".format(l+1)) * Z_norm[l] + session.run("beta{}:0".format(l+1)))   # Batch normalization
	        A.append(np.maximum(Z_tilde[l], 0))                                                                 # Relu activation function
	      else:
	        A.append(np.maximum(Z[l], 0))
	      # Dropout:
	      if dropout_mode==True:
	        np.random.seed((alt+sim))                                                      # Fixing the seed
	        D.append(np.random.rand(A[l].shape[0], A[l].shape[1]) < keep_prob_lst[alt])   # Generating random binary indicators
	        A[l] = np.divide(np.multiply(A[l], D[l]), keep_prob_lst[alt])              # Dropout regularization
	    # Output layer:
	    Y_pred_lst[alt, :, sim] = np.dot(session.run("W_out:0"), A[n_layers-1]) + session.run("B_out:0")   # Initialize linear predictor
	# Adding to the sets list:
	Y_pred_lst_sets.append(Y_pred_lst)


# Updating the number of sets:
n_sets=3

# Initialize a variable to receive metrics:
rmse_sets = np.empty([n_alt, n_sets])
r2_sets = np.empty([n_alt, n_sets])
mic_sets = np.empty([n_alt, n_sets])

# Getting metrics:
for c in range(n_sets):
	# Getting final predictions:
	Y_pred = np.mean(Y_pred_lst_sets[c], axis=2)
	# Computing RMSE of the prediction using the dev set:
	if c==0:
	    y_tmp = y['trn']
	elif c==1:
	    y_tmp = y['dev']
	else:
	    y_tmp = y['tst']
	for alt in range(n_alt):
	    rmse_sets[alt, c] = rmse(y_tmp, Y_pred[alt,:])
	    r2_sets[alt, c] = r2_score(np.squeeze(y_tmp), Y_pred[alt,:])
	    mic_sets[alt, c] = normalized_mutual_info_score(np.squeeze(y_tmp), Y_pred[alt,:])


# Printing RMSE:
print(np.round(np.sort(rmse_sets[:,0], axis=0),4))
print(np.round(np.sort(rmse_sets[:,1], axis=0),4))
print(np.round(np.sort(rmse_sets[:,2], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,0], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,1], axis=0),4))
print(np.round(np.argsort(rmse_sets[:,2], axis=0),4))

# Printing r2:
print(np.round(np.sort(r2_sets[:,0], axis=0)[::-1],4))
print(np.round(np.sort(r2_sets[:,1], axis=0)[::-1],4))
print(np.round(np.sort(r2_sets[:,2], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,0], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,1], axis=0)[::-1],4))
print(np.round(np.argsort(r2_sets[:,2], axis=0)[::-1],4))

# Printing Mutual Information Criteria:
print(np.round(np.sort(mic_sets[:,0], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,1], axis=0)[::-1],4))
print(np.round(np.sort(mic_sets[:,2], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,0], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,1], axis=0)[::-1],4))
print(np.round(np.argsort(mic_sets[:,2], axis=0)[::-1],4))

#--------------------------------------------Submission file-------------------------------------------------#

# # Setting directory:
# os.chdir(prefix_proj + "codes")

# # Loading external functions:
# from external_functions import * 

# # Setting directory where the results were stored:
# os.chdir("/home/jhonathan/Documentos/deep_phd-proj/results")
# # os.chdir("/data1/aafgarci/jhonathan/deep_phd-proj/results")

# # Number of submission
# n_submission = 8

# # Getting the index of the best predictions:
# index = get_index(array=rmse_sets[:,:,2], n=n_submission)

# # Getting the mean across dropout vectors:
# Y_pred = np.mean(Y_pred_lst_sets[3], axis=2)

# # Creating submission file and storing:
# for i, j in zip(index[0], index[1]):
#     print([str(i), str(j)])
#     # Create empty submission dataframe
#     sub = pd.DataFrame()
#     # Insert ID and Predictions into dataframe
#     sub['Id'] = dfTst['Id']
#     sub['SalePrice'] = np.expm1(Y_pred[i, :, j])
#     # Output submission file
#     sub.to_csv("submission_model"+ str(j) + "_alt" + str(i) + ".csv",index=False)

