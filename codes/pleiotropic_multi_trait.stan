
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;
  int<lower=1> p_x_0;
  int<lower=1> p_z;
  int<lower=1> p_i_0;
  int<lower=1> p_r_0;

  // Global hyperparameter:
  real phi_0;

  // Vector for specific priors on each feature:
  int index_x_0[p_x_0];

  // Feature matrices:
  matrix[n_0, p_x_0] X_0;
  matrix[n_0, p_z] Z_0;
  matrix[n_0, p_r_0] X_r_0;
  
  // Phenotypic vector:
  real y_0[n_0];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_1;
  int<lower=1> p_x_1;
  int<lower=1> p_i_1;
  int<lower=1> p_r_1;

  // Global hyperparameter:
  real phi_1;

  // Vector for specific priors on each feature:
  int index_x_1[p_x_1];

  // Feature matrices:
  matrix[n_1, p_x_1] X_1;
  matrix[n_1, p_z] Z_1;
  matrix[n_1, p_r_1] X_r_1;
  
  // Phenotypic vector:
  real y_1[n_1];

}

parameters {

  // Parameters:
  real mu_0;
  vector[p_x_0] beta_0;
  vector[p_z] alpha_0;  
  vector<lower=0>[p_r_0] sigma_0;

  // First level hyperparameters:
  real u_mu_0;
  vector[p_i_0] u_beta_0;
  real u_alpha_0;

  real<lower=0> s_mu_0;
  vector<lower=0>[p_i_0] s_beta_0;
  real<lower=0> s_alpha_0;
  real<lower=0> s_sigma_0;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_0;
  real<lower=0> pi_u_beta_0;
  real<lower=0> pi_u_alpha_0;

  real<lower=0> pi_s_mu_0;
  real<lower=0> pi_s_beta_0;
  real<lower=0> pi_s_alpha_0;
  real<lower=0> pi_s_sigma_0;

  // Defining variable to generate data from the model:
  real y_gen_0[n_0];

  // Parameters:
  real mu_1;
  vector[p_x_1] beta_1;
  vector[p_z] alpha_1;
  vector<lower=0>[p_r_1] sigma_1;

  // First level hyperparameters:
  real u_mu_1;
  vector[p_i_1] u_beta_1;
  real u_alpha_1;

  real<lower=0> s_mu_1;
  vector<lower=0>[p_i_1] s_beta_1;
  real<lower=0> s_alpha_1;
  real<lower=0> s_sigma_1;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_1;
  real<lower=0> pi_u_beta_1;
  real<lower=0> pi_u_alpha_1;

  real<lower=0> pi_s_mu_1;
  real<lower=0> pi_s_beta_1;
  real<lower=0> pi_s_alpha_1;
  real<lower=0> pi_s_sigma_1;

  // Defining variable to generate data from the model:
  real y_gen_1[n_1];

  // Defining variables to create dependency:
  real<lower=0> pi_u_eta;
  real<lower=0> pi_s_eta;
  real<lower=0> s_eta;
  real u_eta;
  vector[p_z] eta;

}

transformed parameters {

  // Declaring variables to receive input:
  vector<lower=0>[n_x_0] sigma_vec_0;
  vector[n_x_0] expectation_0;
  vector<lower=0>[n_x_1] sigma_vec_1;
  vector[n_x_1] expectation_1;

  // Computing the vectorized vector of residuals:
  sigma_vec_0 = X_r_0 * sigma_0;

  // Computing the expectation of the likelihood function:
  expectation_0 = mu_0 + X_0 * beta_0 + Z_0 * (alpha_0 + eta);

  // Computing the vectorized vector of residuals:
  sigma_vec_1 = X_r_1 * sigma_1;

  // Computing the expectation of the likelihood function:
  expectation_1 = mu_1 + X_1 * beta_1 + Z_1 * (alpha_1 + eta);

}

model {

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta ~ cauchy(0, phi_1);
  pi_s_eta ~ cauchy(0, phi_1);
  s_eta ~ cauchy(0, pi_s_eta);
  u_eta ~ normal(0, pi_u_eta);
  eta ~ normal(u_eta, s_eta);

  //// First response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_0 ~ cauchy(0,  phi_0);
  pi_u_beta_0 ~ cauchy(0,  phi_0);
  pi_u_alpha_0 ~ cauchy(0,  phi_0);

  pi_s_mu_0 ~ cauchy(0,  phi_0);
  pi_s_beta_0 ~ cauchy(0,  phi_0);
  pi_s_alpha_0 ~ cauchy(0,  phi_0);
  pi_s_sigma_0 ~ cauchy(0,  phi_0);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_0 ~ normal(0, pi_u_mu_0);
  u_beta_0 ~ normal(0, pi_u_beta_0);
  u_alpha_0 ~ normal(0, pi_u_alpha_0);

  s_mu_0 ~ cauchy(0, pi_s_mu_0);
  s_beta_0 ~ cauchy(0, pi_s_beta_0);
  s_alpha_0 ~ cauchy(0, pi_s_alpha_0);
  s_sigma_0 ~ cauchy(0, pi_s_sigma_0);

  // Specifying priors for the parameters:
  mu_0 ~ normal(u_mu_0, s_mu_0);
  beta_0 ~ normal(u_beta_0[index_x_0], s_beta_0[index_x_0]);
  alpha_0 ~ normal(u_alpha_0, s_alpha_0);
  sigma_0 ~ cauchy(0, s_sigma_0);

  // Specifying the likelihood:
  y_0 ~ normal(expectation_0, sigma_vec_0);

  // Generating data from the model:
  y_gen_0 ~ normal(expectation_0, sigma_vec_0);

  //// Second response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_1 ~ cauchy(0,  phi_1);
  pi_u_beta_1 ~ cauchy(0,  phi_1);
  pi_u_alpha_1 ~ cauchy(0,  phi_1);

  pi_s_mu_1 ~ cauchy(0,  phi_1);
  pi_s_beta_1 ~ cauchy(0,  phi_1);
  pi_s_alpha_1 ~ cauchy(0,  phi_1);
  pi_s_sigma_1 ~ cauchy(0,  phi_1);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_1 ~ normal(0, pi_u_mu_1);
  u_beta_1 ~ normal(0, pi_u_beta_1);
  u_alpha_1 ~ normal(0, pi_u_alpha_1);

  s_mu_1 ~ cauchy(0, pi_s_mu_1);
  s_beta_1 ~ cauchy(0, pi_s_beta_1);
  s_alpha_1 ~ cauchy(0, pi_s_alpha_1);
  s_sigma_1 ~ cauchy(0, pi_s_sigma_1);

  // Specifying priors for the parameters:
  mu_1 ~ normal(u_mu_1, s_mu_1);
  beta_1 ~ normal(u_beta_1[index_x_1], s_beta_1[index_x_1]);
  alpha_1 ~ normal(u_alpha_1, s_alpha_1);
  sigma_1 ~ cauchy(0, s_sigma_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma_vec_1);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma_vec_1);


}
