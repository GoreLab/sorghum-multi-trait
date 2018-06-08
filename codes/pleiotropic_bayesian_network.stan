
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;
  int<lower=1> p_x_0;
  int<lower=1> p_z;
  int<lower=1> p_r_0;

  // Feature matrices:
  matrix[n_0, p_x_0] X_0;
  matrix[n_0, p_z] Z_0;
  matrix[n_0, p_r_0] X_r_0;
  
  // Phenotypic vector:
  real y_0[n_0];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_1;
  int<lower=1> p_x_1;
  int<lower=1> p_r_1;

  // Feature matrices:
  matrix[n_1, p_x_1] X_1;
  matrix[n_1, p_z] Z_1;
  matrix[n_1, p_r_1] X_r_1;
  
  // Phenotypic vector:
  real y_1[n_1];

  // Global known hyperparameter:
  real phi;

}

parameters {

  // Parameters:
  vector[p_x_0] beta_0;
  vector[p_z] alpha_0;  
  vector<lower=0>[p_r_0] sigma_0;

  // First level hyperparameters:
  real u_beta_0;
  real u_alpha_0;

  real<lower=0> s_beta_0;
  real<lower=0> s_alpha_0;
  real<lower=0> s_sigma_0;

  // Second level hyperparameters:
  real<lower=0> pi_u_beta_0;
  real<lower=0> pi_u_alpha_0;

  real<lower=0> pi_s_beta_0;
  real<lower=0> pi_s_alpha_0;
  real<lower=0> pi_s_sigma_0;

  // Defining variable to generate data from the model:
  real y_gen_0[n_0];

  // Parameters:
  vector[p_x_1] beta_1;
  vector[p_z] alpha_1;
  vector<lower=0>[p_r_1] sigma_1;

  // First level hyperparameters:
  real u_beta_1;
  real u_alpha_1;

  real<lower=0> s_beta_1;
  real<lower=0> s_alpha_1;
  real<lower=0> s_sigma_1;

  // Second level hyperparameters:
  real<lower=0> pi_u_beta_1;
  real<lower=0> pi_u_alpha_1;

  real<lower=0> pi_s_beta_1;
  real<lower=0> pi_s_alpha_1;
  real<lower=0> pi_s_sigma_1;

  // Defining variable to generate data from the model:
  real y_gen_1[n_1];

  // Defining variables to create dependency (1st variable):
  real<lower=0> pi_u_z_0;
  real<lower=0> pi_s_z_0;
  real<lower=0> s_z_0;
  real u_z_0; 

  // Defining variables to create dependency (2nd variable):
  real<lower=0> pi_u_z_1;
  real<lower=0> pi_s_z_1;
  real<lower=0> s_z_1;
  real u_z_1; 

  // Defining global pleiotropic effect:
  vector[p_z] z;

}

transformed parameters {

  // Declaring variables to receive input:
  vector[p_z] eta_0;
  vector[p_z] eta_1;
  vector<lower=0>[n_0] sigma_vec_0;
  vector[n_0] expectation_0;
  vector<lower=0>[n_1] sigma_vec_1;
  vector[n_1] expectation_1;

  // Computing pleotropic effect:
  eta_0 = u_z_0 + s_z_0 * z;

  // Computing the vectorized vector of residuals:
  sigma_vec_0 = X_r_0 * sigma_0;

  // Computing the expectation of the likelihood function:
  expectation_0 = X_0 * beta_0 + Z_0 * (alpha_0 + eta_0);

  // Computing pleotropic effect:
  eta_1 = u_z_1 + s_z_1 * z;

  // Computing the vectorized vector of residuals:
  sigma_vec_1 = X_r_1 * sigma_1;

  // Computing the expectation of the likelihood function:
  expectation_1 = X_1 * beta_1 + Z_1 * (alpha_1 + eta_1);

}

model {

  //// Conditional probabilities distributions that creates dependecy between the responses:

  // Pleiotropy modularity hyperparameters:
  pi_u_z_0 ~ cauchy(0, phi);
  pi_s_z_0 ~ cauchy(0, phi);
  s_z_0 ~ cauchy(0, pi_s_z_0);
  u_z_0 ~ normal(0, pi_u_z_0);

  pi_u_z_1 ~ cauchy(0, phi);
  pi_s_z_1 ~ cauchy(0, phi);
  s_z_1 ~ cauchy(0, pi_s_z_1);
  u_z_1 ~ normal(0, pi_u_z_1);

  // Pleiotropic tying parameter:
  z ~ normal(0, 1);

  //// First response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_beta_0 ~ cauchy(0, phi);
  pi_u_alpha_0 ~ cauchy(0, phi);

  pi_s_beta_0 ~ cauchy(0, phi);
  pi_s_alpha_0 ~ cauchy(0, phi);
  pi_s_sigma_0 ~ cauchy(0, phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_beta_0 ~ normal(0, pi_u_beta_0);
  u_alpha_0 ~ normal(0, pi_u_alpha_0);

  s_beta_0 ~ cauchy(0, pi_s_beta_0);
  s_alpha_0 ~ cauchy(0, pi_s_alpha_0);
  s_sigma_0 ~ cauchy(0, pi_s_sigma_0);

  // Specifying priors for the parameters:
  beta_0 ~ normal(u_beta_0, s_beta_0);
  alpha_0 ~ normal(u_alpha_0, s_alpha_0);
  sigma_0 ~ cauchy(0, s_sigma_0);

  // Specifying the likelihood:
  y_0 ~ normal(expectation_0, sigma_vec_0);

  // Generating data from the model:
  y_gen_0 ~ normal(expectation_0, sigma_vec_0);

  //// Second response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_beta_1 ~ cauchy(0, phi);
  pi_u_alpha_1 ~ cauchy(0, phi);

  pi_s_beta_1 ~ cauchy(0, phi);
  pi_s_alpha_1 ~ cauchy(0, phi);
  pi_s_sigma_1 ~ cauchy(0, phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_beta_1 ~ normal(0, pi_u_beta_1);
  u_alpha_1 ~ normal(0, pi_u_alpha_1);

  s_beta_1 ~ cauchy(0, pi_s_beta_1);
  s_alpha_1 ~ cauchy(0, pi_s_alpha_1);
  s_sigma_1 ~ cauchy(0, pi_s_sigma_1);

  // Specifying priors for the parameters:
  beta_1 ~ normal(u_beta_1, s_beta_1);
  alpha_1 ~ normal(u_alpha_1, s_alpha_1);
  sigma_1 ~ cauchy(0, s_sigma_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma_vec_1);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma_vec_1);


}
