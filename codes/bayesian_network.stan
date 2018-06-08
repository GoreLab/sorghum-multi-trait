
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n;
  int<lower=1> p_x;
  int<lower=1> p_z;
  int<lower=1> p_r;

  // Feature matrices:
  matrix[n, p_x] X;
  matrix[n, p_z] Z;
  matrix[n, p_r] X_r;
  
  // Phenotypic vector:
  real y[n];

  // Global known hyperparameter:
  real phi;

}

parameters {

  // Parameters:
  vector[p_x] beta;
  vector[p_z] alpha;  
  vector<lower=0>[p_r] sigma;

  // First level hyperparameters:
  real u_beta;
  real u_alpha;

  real<lower=0> s_beta;
  real<lower=0> s_alpha;
  real<lower=0> s_sigma;

  // Second level hyperparameters:
  real<lower=0> pi_u_beta;
  real<lower=0> pi_u_alpha;

  real<lower=0> pi_s_beta;
  real<lower=0> pi_s_alpha;
  real<lower=0> pi_s_sigma;

  // Defining variable to generate data from the model:
  real y_gen[n];

}

transformed parameters {

  // Declaring variables to receive input:
  vector<lower=0>[n] sigma_vec;
  vector[n] expectation;

  // Computing the vectorized vector of residuals:
  sigma_vec = X_r * sigma;

  // Computing the expectation of the likelihood function:
  expectation = X * beta + Z * alpha;

}

model {

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_beta ~ cauchy(0,  phi);
  pi_u_alpha ~ cauchy(0,  phi);

  pi_s_beta ~ cauchy(0,  phi);
  pi_s_alpha ~ cauchy(0,  phi);
  pi_s_sigma ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_beta ~ normal(0, pi_u_beta);
  u_alpha ~ normal(0, pi_u_alpha);

  s_beta ~ cauchy(0, pi_s_beta);
  s_alpha ~ cauchy(0, pi_s_alpha);
  s_sigma ~ cauchy(0, pi_s_sigma);

  // Specifying priors for the parameters:
  beta ~ normal(u_beta, s_beta);
  alpha ~ normal(u_alpha, s_alpha);
  sigma ~ cauchy(0, s_sigma);

  // Specifying the likelihood:
  y ~ normal(expectation, sigma_vec);

  // Generating data from the model:
  y_gen ~ normal(expectation, sigma_vec);

}
