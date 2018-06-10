
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;
  int<lower=1> p_z;

  // Feature matrices:
  matrix[n_0, p_z] Z_0;
  
  // Phenotypic vector:
  real y_0[n_0];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_1;

  // Feature matrices:
  matrix[n_1, p_z] Z_1;
  
  // Phenotypic vector:
  real y_1[n_1];

  // Global known hyperparameter:
  real phi;

}

parameters {

  // Population mean parameter/hyperparameters:
  real<lower=0> pi_s_mu_0;
  real<lower=0> s_mu_0;
  real mu_0;

  // Features parameter/hyperparameters:
  real<lower=0> pi_s_alpha_0;
  real<lower=0> s_alpha_0;
  vector[p_z] alpha_0;  

  // Residual parameter/hyperparameters:
  real<lower=0> pi_s_sigma_0;
  real<lower=0> s_sigma_0;
  real<lower=0> sigma_0;

  // Defining variable to generate data from the model:
  real y_gen_0[n];

  // Population mean parameter/hyperparameters:
  real<lower=0> pi_s_mu_1;
  real<lower=0> s_mu_1;
  real mu_1;

  // Features parameter/hyperparameters:
  real<lower=0> pi_s_alpha_1;
  real<lower=0> s_alpha_1;
  vector[p_z] alpha_1;  

  // Residual parameter/hyperparameters:
  real<lower=0> pi_s_sigma_1;
  real<lower=0> s_sigma_1;
  real<lower=0> sigma_1;

  // Defining variable to generate data from the model:
  real y_gen_1[n];

  // Defining variables to create dependency (1st variable):
  real<lower=0> pi_s_sigma_z_0;
  real<lower=0> s_sigma_z_0;
  real<lower=0> sigma_z_0;

  real<lower=0> pi_s_mu_z_0;
  real<lower=0> s_mu_z_0;
  real<lower=0> mu_z_0;

  // Defining variables to create dependency (2nd variable):
  real<lower=0> pi_s_sigma_z_1;
  real<lower=0> s_sigma_z_1;
  real<lower=0> sigma_z_1;

  real<lower=0> pi_s_mu_z_1;
  real<lower=0> s_mu_z_1;
  real<lower=0> mu_z_1;

  // Defining global pleiotropic effect:
  vector[p_z] z;

}

transformed parameters {

  // Declaring variables to receive input:
  vector[p_z] eta_0;
  vector[p_z] eta_1;
  vector[n_0] expectation_0;
  vector[n_1] expectation_1;

  // Computing pleotropic effect:
  eta_0 = mu_z_0 + sigma_z_0 * z;

  // Computing the expectation of the likelihood function:
  expectation_0 = mu_0 + Z_0 * (alpha_0 + eta_0);

  // Computing pleotropic effect:
  eta_1 = mu_z_1 + sigma_z_1 * z;

  // Computing the expectation of the likelihood function:
  expectation_1 = mu_1 + Z_1 * (alpha_1 + eta_1);

}

model {

  //// Conditional probabilities distributions that creates dependecy between the responses:

  // Pleiotropy modularity hyperparameters:
  pi_s_mu_z_0 ~ cauchy(0, phi);
  s_mu_z_0 ~ cauchy(0, pi_s_mu_z_0);
  mu_z_0 ~ normal(0, s_mu_z_0);

  pi_s_sigma_z_0 ~ cauchy(0, phi);
  s_sigma_z_0 ~ cauchy(0, pi_s_sigma_z_0);
  sigma_z_0 ~ normal(0, s_sigma_z_0);

  pi_s_mu_z_1 ~ cauchy(0, phi);
  s_mu_z_1 ~ cauchy(0, pi_s_mu_z_1);
  mu_z_1 ~ normal(0, s_mu_z_1);

  pi_s_sigma_z_1 ~ cauchy(0, phi);
  s_sigma_z_1 ~ cauchy(0, pi_s_sigma_z_1);
  sigma_z_1 ~ normal(0, s_sigma_z_1);

  // Pleiotropic tying parameter:
  z ~ normal(0, 1);

  //// First response variable conditionals probability distributions:

  //// Conditional probabilities distributions for mean:
  pi_s_mu_0 ~ cauchy(0,  phi);
  s_mu_0 ~ cauchy(0, pi_s_mu_0);
  mu_0 ~ normal(0, s_mu_0);

  //// Conditional probabilities distributions for features:
  pi_s_alpha_0 ~ cauchy(0,  phi);
  s_alpha_0 ~ cauchy(0, pi_s_alpha_0);
  alpha_0 ~ normal(0, s_alpha_0);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma_0 ~ cauchy(0,  phi);
  s_sigma_0 ~ cauchy(0, pi_s_sigma_0);
  sigma_0 ~ cauchy(0, s_sigma_0);

  // Specifying the likelihood:
  y_0 ~ normal(expectation_0, sigma_0);

  // Generating data from the model:
  y_gen_0 ~ normal(expectation_0, sigma_0);

  //// Second response variable conditionals probability distributions:

  //// Conditional probabilities distributions for mean:
  pi_s_mu_1 ~ cauchy(0,  phi);
  s_mu_1 ~ cauchy(0, pi_s_mu_1);
  mu_1 ~ normal(0, s_mu_1);

  //// Conditional probabilities distributions for features:
  pi_s_alpha_1 ~ cauchy(0,  phi);
  s_alpha_1 ~ cauchy(0, pi_s_alpha_1);
  alpha_1 ~ normal(0, s_alpha_1);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma_1 ~ cauchy(0,  phi);
  s_sigma_1 ~ cauchy(0, pi_s_sigma_1);
  sigma_1 ~ cauchy(0, s_sigma_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma_1);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma_1);

}
