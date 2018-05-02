
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_x;
  int<lower=1> p_x;
  int<lower=1> p_r;

  // Global hyperparameter:
  real phi;

  // Vector for specific priors on each feature:
  int index_x[p_x];

  // Feature matrices:
  matrix[n_x, p_x] X;
  matrix[n_x, p_r] X_r;
  
  // Phenotypic vector:
  real y[n_x];

}

parameters {

  // Parameters:
  real mu;
  vector[p_x] beta;
  vector<lower=0>[p_r] sigma;

  // First level hyperparameters:
  real u_mu;
  vector[p_x] u_beta;
  vector[p_r] u_sigma;

  real<lower=0> s_mu;
  vector<lower=0>[p_x] s_beta;
  vector<lower=0>[p_r] s_sigma;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu;
  real<lower=0> pi_u_beta;
  real<lower=0> pi_u_sigma;

  real<lower=0> pi_s_mu;
  real<lower=0> pi_s_beta;
  real<lower=0> pi_s_sigma;

  // Defining variable to generate data from the model:
  real y_rep[n_x];

}

transformed parameters {

  // Declaring variables to receive input:
  vector<lower=0>[n_x] sigma_vec;
  vector[n_x] expectation;

  // Computing the vectorized vector of residuals:
  sigma_vec = X_r * sigma;

  // Computing the expectation of the likelihood function:
  expectation = X * beta;

}

model {

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu ~ cauchy(0,  phi);
  pi_u_beta ~ cauchy(0,  phi);
  pi_u_sigma ~ cauchy(0,  phi);

  pi_s_mu ~ cauchy(0,  phi);
  pi_s_beta ~ cauchy(0,  phi);
  pi_s_sigma ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu ~ normal(0, pi_u_mu);
  u_beta ~ normal(0, pi_u_beta);
  u_sigma ~ normal(0, pi_u_sigma);

  s_mu ~ cauchy(0, pi_s_mu);
  s_beta ~ cauchy(0, pi_s_beta);
  s_sigma ~ cauchy(0, pi_s_sigma);

  // Specifying priors for the parameters:
  mu ~ normal(u_mu, s_mu);
  beta ~ normal(u_beta[index_x], s_sigma[index_x]);
  sigma ~ cauchy(u_sigma, s_sigma);

  // Specifying the likelihood:
  y ~ normal(expectation, sigma_vec);

  // Generating data from the model:
  y_rep ~ normal(expectation, sigma_vec);

}
