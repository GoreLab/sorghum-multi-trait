



#----------------------------------------Example of a python class-------------------------------------------#

class Kls(object):
    def __init__(self, data):
        self.data = data
 
    def printd(self):
        print(self.data)
 
ik1 = Kls('arun')
ik2 = Kls('seema')
 
ik1.printd()
ik2.printd()


#----------------------------------------Test via ridge regression-------------------------------------------#

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#transform y_train to match the evaluation metric
y_train=y['trn'].transpose()

#create X_train and X_test
X_train=X['trn'].transpose()
X_test=X['dev'].transpose()

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# steps
steps = [('scaler', StandardScaler()),
         ('ridge', Ridge())]
# steps = [('scaler', StandardScaler()),
#          ('lasso', Lasso())]


# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {('ridge__alpha'):np.logspace(-4, 0, 100)}

# Create the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

#predict on train set
y_pred_train=cv.predict(X_train)

# Predict test set
y_pred_test=cv.predict(X_test)

# rmse on train set
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Root Mean Squared Error: {}".format(rmse))

# rmse on test set
rmse = np.sqrt(mean_squared_error(y['dev'].transpose(), y_pred_test))
print("Root Mean Squared Error: {}".format(rmse))

# Plotting:
y_tmp = dict()
y_tmp['trn'] = y_pred_train
y_tmp['dev'] = y_pred_test
plt.scatter(y['trn'], y_tmp['trn'], color='red')
plt.scatter(y['dev'], y_tmp['dev'], color='green')
plt.xlim(2.5, 4)
plt.ylim(2.5, 4)
plt.title('Observed vs predicted data')
plt.xlabel('Observed transcription binned values')
plt.ylabel("Predicted transcription binned values")
plt.show()


#--------------------------------------To run the code on pystan---------------------------------------------#

import pystan as ps

# Transpose:
X['trn'] = X['trn'].transpose()
X['dev'] = X['dev'].transpose()
X['tst'] = X['tst'].transpose()
y['trn'] = y['trn'].transpose()
y['dev'] = y['dev'].transpose()
y['tst'] = y['tst'].transpose()

# Getting the features names prefix:
tmp = X['trn'].columns.str.split('_').str.get(0)

# Building an incidence vector for adding specific priors for each feature class:
index_x = pd.DataFrame(tmp).replace(tmp.drop_duplicates(), range(1,(tmp.drop_duplicates().size+1)))[0].values 

# Building an year matrix just for indexing resuduals standard deviations heterogeneous across time:
X['year'] = np.ones(y['trn'].shape) 

# # For subsetting for tests:
# subset1 = np.random.choice(range(X['trn'].shape[0]), size=100)
# subset2 = X['trn'].index[subset1]

# # Storing all the data into a dictionary for pystan:
# df_stan = dict(n_x = X['trn'].loc[subset2,:].shape[0],
# 			   p_x = X['trn'].shape[1],
# 			   p_i = np.max(index_x),
# 			   p_r = X['year'].shape[1],
# 			   phi = np.max(y['trn'][subset1])*10,
# 			   index_x = index_x,
# 			   X = X['trn'].loc[subset2,:],
# 			   X_r = X['year'].loc[subset2,:],
# 			   y = y['trn'][subset1].reshape((y['trn'][subset1].shape[0],)))

# Storing all the data into a dictionary for pystan:
df_stan = dict(n_x = X['trn'].shape[0],
			   p_x = X['trn'].shape[1],
			   p_i = np.max(index_x),
			   p_r = X['year'].shape[1],
			   phi = np.max(y['trn'])*10,
			   index_x = index_x,
			   X = X['trn'],
			   X_r = X['year'],
			   y = y['trn'].flatten())

# Setting directory:
os.chdir(prefix_proj + "codes")

# Compiling the C++ code for the model:
model = ps.StanModel(file='multi_trait.stan')

# Creating an empty dict:
fit = dict()

# Fitting the model:
fit['400'] = model.sampling(data=df_stan, chains=1, iter=400)

# Getting posterior means:
beta_mean = dict()
mu_mean = dict()
beta_mean['400'] = fit['400'].extract()['beta'].mean(axis=0)
mu_mean['400'] = fit['400'].extract()['mu'].mean(axis=0)

# Computing predictions for trn:
y_pred = dict()
y_pred['trn'] = mu_mean['400'] + X['trn'].dot(beta_mean['400'])

# Printing train rMSE errors:
rmse(y['trn'].flatten(), y_pred['trn'])

# Computing predictions for dev:
y_pred['dev'] = mu_mean['400'] + X['dev'].dot(beta_mean['400'])

# Printing dev rMSE errors:
rmse(y['dev'].flatten(), y_pred['dev'])

# Computing predictions for test:
y_pred['tst'] = mu_mean['400'] + X['tst'].dot(beta_mean['400'])

# Printing test rMSE errors:
rmse(y['tst'].flatten(), y_pred['tst'])

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

# Plots of the observed against the generated:
sns.set_style('whitegrid')
ax = sns.kdeplot(fit['400'].extract()['y_rep'].mean(axis=0), bw=0.5, label='1_400', shade=True)
ax = sns.kdeplot(y['trn'].flatten(), bw=0.5, label='obs', shade=True)
ax.set_title('Observed vs generated data (nchain_niter)')
ax.set(xlabel='Dry mass values', ylabel='Density')
plt.show()
plt.savefig(prefix_out + 'plots/' + 'biomass_iter_tunning_obs_gen' + '.pdf')
plt.clf()

# Plotting:
plt.scatter(y['trn'], y_pred['trn'], color='red')
plt.scatter(y['dev'], y_pred['dev'], color='green')
plt.scatter(y['tst'], y_pred['tst'], color='blue')
# plt.xlim(2.5, 4)
# plt.ylim(2.5, 4)
plt.title('Observed vs predicted data')
plt.xlabel('Observed transcription binned values')
plt.ylabel("Predicted transcription binned values")
plt.show()
