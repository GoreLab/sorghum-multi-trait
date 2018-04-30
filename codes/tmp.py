
#------------------------------------------Development section-----------------------------------------------#

# To reset the previous graph:
tf.reset_default_graph()

# Number of simulations:
if dropout_mode==True:
	n_sim=100

# If there is no drop out:
if dropout_mode==False:
	n_sim=1

# Small epsilon value for batch norm
epsilon = 1e-7

# Number of data sets:
n_sets = 4

# Initializing lists:
Y_pred_lst_sets = []

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
	# For hidden set:
	if c==3:
	  X_tmp = X['miss']
	# Initializing list:
	Y_pred_lst = np.empty(shape=[n_alt, X_tmp.shape[1] , n_sim])
	for alt in range(n_alt):
	  # Importing results:
	  tf.reset_default_graph()
	  n_layers = n_layers_lst[alt]
	  session = tf.Session() 
	  save_path = prefix_out + "outputs/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + "/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r) + ".meta"
	  saver = tf.train.import_meta_graph(save_path, clear_devices=True)
	  save_path = prefix_out + "outputs/core" + str(core) + "_alt" + str(alt) + "_bin" + str(r)
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
	        np.random.seed(seed)                                                      # Fixing the seed
	        D.append(np.random.rand(A[l].shape[0], A[l].shape[1]) < keep_prob_lst[alt])   # Generating random binary indicators
	        A[l] = np.divide(np.multiply(A[l], D[l]), keep_prob_lst[alt])              # Dropout regularization
	    # Output layer:
	    Y_pred_lst[alt, :, sim] = np.dot(session.run("W_out:0"), A[n_layers-1]) + session.run("B_out:0")   # Initialize linear predictor
	# Adding to the sets list:
	Y_pred_lst_sets.append(Y_pred_lst)




# Updating the number of sets:
n_sets=3

# Initialize a variable to receive metrics:
rmse_sets = np.empty([n_alt, n_core, n_sets])
cor_sets = np.empty([n_alt, n_core, n_sets])
mic_sets = np.empty([n_alt, n_core, n_sets])

# Getting metrics:
for c in range(n_sets):
  # Getting final predictions:
  Y_pred = np.mean(Y_pred_lst_sets[c], axis=2)
  # Computing RMSE of the prediction using the dev set:
  for m in range(n_core):
    # Loading data:
    container = np.load("out_core" + str(core) + ".npz")
    data = [container[key] for key in container]
    y_trn = data[0][0]
    y_dev = data[0][2]
    y_tst = data[0][4]
    # Getting data set:
    if c==0:
        y = y_trn
    elif c==1:
        y = y_dev
    else:
        y = y_tst
    for alt in range(n_alt):
        rmse_sets[alt, m, c] = rmse(y[alt], Y_pred[alt,:,m])
        cor_sets[alt, m, c] = pearsonr(np.squeeze(y[alt]), Y_pred[alt,:,m])[0]
        mic_sets[alt, m, c] = normalized_mutual_info_score(np.squeeze(y[alt]), Y_pred[alt,:,m])

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

# # Changing shape:
# rmse_trn = np.reshape(rmse_trn, [n_alt*n_core,])
# rmse_dev = np.reshape(rmse_dev, [n_alt*n_core,])
# rmse_tst = np.reshape(rmse_tst, [n_alt*n_core,])

