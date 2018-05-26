
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;
  int<lower=1> p_x_0;
  int<lower=1> p_z;
  int<lower=1> p_i_0;
  int<lower=1> p_r_0;

  // Vector for specific priors on each feature:
  int index_x_0[p_x_0];

  // Feature matrices:
  vector[n_0] x_d_0;
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

  // Vector for specific priors on each feature:
  int index_x_1[p_x_1];

  // Feature matrices:
  vector[n_1] x_d_1;
  matrix[n_1, p_x_1] X_1;
  matrix[n_1, p_z] Z_1;
  matrix[n_1, p_r_1] X_r_1;

  // Phenotypic vector:
  real y_1[n_1];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_2;
  int<lower=1> p_x_2;
  int<lower=1> p_i_2;
  int<lower=1> p_r_2;

  // Vector for specific priors on each feature:
  int index_x_2[p_x_2];

  // Feature matrices:
  vector[n_2] x_d_2;
  matrix[n_2, p_x_2] X_2;
  matrix[n_2, p_z] Z_2;
  matrix[n_2, p_r_2] X_r_2;

  // Phenotypic vector:
  real y_2[n_2];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_3;
  int<lower=1> p_x_3;
  int<lower=1> p_i_3;
  int<lower=1> p_r_3;

  // Vector for specific priors on each feature:
  int index_x_3[p_x_3];

  // Feature matrices:
  vector[n_3] x_d_3;
  matrix[n_3, p_x_3] X_3;
  matrix[n_3, p_z] Z_3;
  matrix[n_3, p_r_3] X_r_3;

  // Phenotypic vector:
  real y_3[n_3];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_4;
  int<lower=1> p_x_4;
  int<lower=1> p_i_4;
  int<lower=1> p_r_4;

  // Vector for specific priors on each feature:
  int index_x_4[p_x_4];

  // Feature matrices:
  vector[n_4] x_d_4;
  matrix[n_4, p_x_4] X_4;
  matrix[n_4, p_z] Z_4;
  matrix[n_4, p_r_4] X_r_4;

  // Phenotypic vector:
  real y_4[n_4];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_5;
  int<lower=1> p_x_5;
  int<lower=1> p_i_5;
  int<lower=1> p_r_5;

  // Vector for specific priors on each feature:
  int index_x_5[p_x_5];

  // Feature matrices:
  vector[n_5] x_d_5;
  matrix[n_5, p_x_5] X_5;
  matrix[n_5, p_z] Z_5;
  matrix[n_5, p_r_5] X_r_5;

  // Phenotypic vector:
  real y_5[n_5];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_6;
  int<lower=1> p_x_6;
  int<lower=1> p_i_6;
  int<lower=1> p_r_6;

  // Vector for specific priors on each feature:
  int index_x_6[p_x_6];

  // Feature matrices:
  vector[n_6] x_d_6;
  matrix[n_6, p_x_6] X_6;
  matrix[n_6, p_z] Z_6;
  matrix[n_6, p_r_6] X_r_6;

  // Phenotypic vector:
  real y_6[n_6];

  // Global hyperparameter:
  real phi;

}

parameters {

  // Global dap effect hyperparameter:
  real<lower=0> pi_u_d;
  real<lower=0> pi_s_d;
  real<lower=0> s_d;
  real u_d;
  real d;

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

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_0_1;
  real<lower=0> pi_s_eta_0_1;
  real<lower=0> s_eta_0_1;
  real u_eta_0_1;
  vector[p_z] eta_0_1;

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

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_1_2;
  real<lower=0> pi_s_eta_1_2;
  real<lower=0> s_eta_1_2;
  real u_eta_1_2;
  vector[p_z] eta_1_2;

  // Parameters:
  real mu_2;
  vector[p_x_2] beta_2;
  vector[p_z] alpha_2;
  vector<lower=0>[p_r_2] sigma_2;

  // First level hyperparameters:
  real u_mu_2;
  vector[p_i_2] u_beta_2;
  real u_alpha_2;

  real<lower=0> s_mu_2;
  vector<lower=0>[p_i_2] s_beta_2;
  real<lower=0> s_alpha_2;
  real<lower=0> s_sigma_2;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_2;
  real<lower=0> pi_u_beta_2;
  real<lower=0> pi_u_alpha_2;

  real<lower=0> pi_s_mu_2;
  real<lower=0> pi_s_beta_2;
  real<lower=0> pi_s_alpha_2;
  real<lower=0> pi_s_sigma_2;

  // Defining variable to generate data from the model:
  real y_gen_2[n_2];

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_2_3;
  real<lower=0> pi_s_eta_2_3;
  real<lower=0> s_eta_2_3;
  real u_eta_2_3;
  vector[p_z] eta_2_3;

  // Parameters:
  real mu_3;
  vector[p_x_3] beta_3;
  vector[p_z] alpha_3;
  vector<lower=0>[p_r_3] sigma_3;

  // First level hyperparameters:
  real u_mu_3;
  vector[p_i_3] u_beta_3;
  real u_alpha_3;

  real<lower=0> s_mu_3;
  vector<lower=0>[p_i_3] s_beta_3;
  real<lower=0> s_alpha_3;
  real<lower=0> s_sigma_3;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_3;
  real<lower=0> pi_u_beta_3;
  real<lower=0> pi_u_alpha_3;

  real<lower=0> pi_s_mu_3;
  real<lower=0> pi_s_beta_3;
  real<lower=0> pi_s_alpha_3;
  real<lower=0> pi_s_sigma_3;

  // Defining variable to generate data from the model:
  real y_gen_3[n_3];

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_3_4;
  real<lower=0> pi_s_eta_3_4;
  real<lower=0> s_eta_3_4;
  real u_eta_3_4;
  vector[p_z] eta_3_4;

  // Parameters:
  real mu_4;
  vector[p_x_4] beta_4;
  vector[p_z] alpha_4;
  vector<lower=0>[p_r_4] sigma_4;

  // First level hyperparameters:
  real u_mu_4;
  vector[p_i_4] u_beta_4;
  real u_alpha_4;

  real<lower=0> s_mu_4;
  vector<lower=0>[p_i_4] s_beta_4;
  real<lower=0> s_alpha_4;
  real<lower=0> s_sigma_4;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_4;
  real<lower=0> pi_u_beta_4;
  real<lower=0> pi_u_alpha_4;

  real<lower=0> pi_s_mu_4;
  real<lower=0> pi_s_beta_4;
  real<lower=0> pi_s_alpha_4;
  real<lower=0> pi_s_sigma_4;

  // Defining variable to generate data from the model:
  real y_gen_4[n_4];

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_4_5;
  real<lower=0> pi_s_eta_4_5;
  real<lower=0> s_eta_4_5;
  real u_eta_4_5;
  vector[p_z] eta_4_5;

  // Parameters:
  real mu_5;
  vector[p_x_5] beta_5;
  vector[p_z] alpha_5;
  vector<lower=0>[p_r_5] sigma_5;

  // First level hyperparameters:
  real u_mu_5;
  vector[p_i_5] u_beta_5;
  real u_alpha_5;

  real<lower=0> s_mu_5;
  vector<lower=0>[p_i_5] s_beta_5;
  real<lower=0> s_alpha_5;
  real<lower=0> s_sigma_5;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_5;
  real<lower=0> pi_u_beta_5;
  real<lower=0> pi_u_alpha_5;

  real<lower=0> pi_s_mu_5;
  real<lower=0> pi_s_beta_5;
  real<lower=0> pi_s_alpha_5;
  real<lower=0> pi_s_sigma_5;

  // Defining variable to generate data from the model:
  real y_gen_5[n_5];

  // Temporal Genetic effect:
  real<lower=0> pi_u_eta_5_6;
  real<lower=0> pi_s_eta_5_6;
  real<lower=0> s_eta_5_6;
  real u_eta_5_6;
  vector[p_z] eta_5_6;

  // Parameters:
  real mu_6;
  vector[p_x_6] beta_6;
  vector[p_z] alpha_6;
  vector<lower=0>[p_r_6] sigma_6;

  // First level hyperparameters:
  real u_mu_6;
  vector[p_i_6] u_beta_6;
  real u_alpha_6;

  real<lower=0> s_mu_6;
  vector<lower=0>[p_i_6] s_beta_6;
  real<lower=0> s_alpha_6;
  real<lower=0> s_sigma_6;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_6;
  real<lower=0> pi_u_beta_6;
  real<lower=0> pi_u_alpha_6;

  real<lower=0> pi_s_mu_6;
  real<lower=0> pi_s_beta_6;
  real<lower=0> pi_s_alpha_6;
  real<lower=0> pi_s_sigma_6;

  // Defining variable to generate data from the model:
  real y_gen_6[n_6];

}

transformed parameters {

  // Declaring variables to receive input:
  vector<lower=0>[n_0] sigma_vec_0;
  vector[n_0] expectation_0;
  vector<lower=0>[n_1] sigma_vec_1;
  vector[n_1] expectation_1;
  vector<lower=0>[n_2] sigma_vec_2;
  vector[n_2] expectation_2;
  vector<lower=0>[n_3] sigma_vec_3;
  vector[n_3] expectation_3;
  vector<lower=0>[n_4] sigma_vec_4;
  vector[n_4] expectation_4;
  vector<lower=0>[n_5] sigma_vec_5;
  vector[n_5] expectation_5;
  vector<lower=0>[n_6] sigma_vec_6;
  vector[n_6] expectation_6;

  // Computing the vectorized vector of residuals:
  sigma_vec_0 = X_r_0 * sigma_0;

  // Computing the expectation of the likelihood function:
  expectation_0 = mu_0 + x_d_0 * d + X_0 * beta_0 + Z_0 * (alpha_0 + eta_0_1);

  // Computing the vectorized vector of residuals:
  sigma_vec_1 = X_r_1 * sigma_1;

  // Computing the expectation of the likelihood function:
  expectation_1 = mu_1 + x_d_1 * d + X_1 * beta_1 + Z_1 * (alpha_1 + eta_1_2);

  // Computing the vectorized vector of residuals:
  sigma_vec_2 = X_r_2 * sigma_2;

  // Computing the expectation of the likelihood function:
  expectation_2 = mu_2 + x_d_2 * d + X_2 * beta_2 + Z_2 * (alpha_2 + eta_2_3);

  // Computing the vectorized vector of residuals:
  sigma_vec_3 = X_r_3 * sigma_3;

  // Computing the expectation of the likelihood function:
  expectation_3 = mu_3 + x_d_3 * d + X_3 * beta_3 + Z_3 * (alpha_3 + eta_3_4);

  // Computing the vectorized vector of residuals:
  sigma_vec_4 = X_r_4 * sigma_4;

  // Computing the expectation of the likelihood function:
  expectation_4 = mu_4 + x_d_4 * d + X_4 * beta_4 + Z_4 * (alpha_4 + eta_4_5);

  // Computing the vectorized vector of residuals:
  sigma_vec_5 = X_r_5 * sigma_5;

  // Computing the expectation of the likelihood function:
  expectation_5 = mu_5 + x_d_5 * d + X_5 * beta_5 + Z_5 * (alpha_5 + eta_5_6);

  // Computing the vectorized vector of residuals:
  sigma_vec_6 = X_r_6 * sigma_6;

  // Computing the expectation of the likelihood function:
  expectation_6 = mu_6 + x_d_6 * d + X_6 * beta_6 + Z_6 * (alpha_6);

}

model {

  //// First response variable conditionals probability distributions:

  //// Conditional probabilities distributions that creates dependecy between the responses across time:
  pi_u_d ~ cauchy(0, phi);
  pi_s_d ~ cauchy(0, phi);
  s_d ~ cauchy(0, pi_s_d);
  u_d ~ normal(0, pi_u_d);
  d ~ normal(u_d, s_d);

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_0 ~ cauchy(0,  phi);
  pi_u_beta_0 ~ cauchy(0,  phi);
  pi_u_alpha_0 ~ cauchy(0,  phi);

  pi_s_mu_0 ~ cauchy(0,  phi);
  pi_s_beta_0 ~ cauchy(0,  phi);
  pi_s_alpha_0 ~ cauchy(0,  phi);
  pi_s_sigma_0 ~ cauchy(0,  phi);

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

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_0_1 ~ cauchy(0, phi);
  pi_s_eta_0_1 ~ cauchy(0, phi);
  s_eta_0_1 ~ cauchy(0, pi_s_eta_0_1);
  u_eta_0_1 ~ normal(0, pi_u_eta_0_1);
  eta_0_1 ~ normal(u_eta_0_1, s_eta_0_1);

  //// Second response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_1 ~ cauchy(0,  phi);
  pi_u_beta_1 ~ cauchy(0,  phi);
  pi_u_alpha_1 ~ cauchy(0,  phi);

  pi_s_mu_1 ~ cauchy(0,  phi);
  pi_s_beta_1 ~ cauchy(0,  phi);
  pi_s_alpha_1 ~ cauchy(0,  phi);
  pi_s_sigma_1 ~ cauchy(0,  phi);

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

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_1_2 ~ cauchy(0, phi);
  pi_s_eta_1_2 ~ cauchy(0, phi);
  s_eta_1_2 ~ cauchy(0, pi_s_eta_1_2);
  u_eta_1_2 ~ normal(0, pi_u_eta_1_2);
  eta_1_2 ~ normal(u_eta_1_2, s_eta_1_2);

  //// Third response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_2 ~ cauchy(0,  phi);
  pi_u_beta_2 ~ cauchy(0,  phi);
  pi_u_alpha_2 ~ cauchy(0,  phi);

  pi_s_mu_2 ~ cauchy(0,  phi);
  pi_s_beta_2 ~ cauchy(0,  phi);
  pi_s_alpha_2 ~ cauchy(0,  phi);
  pi_s_sigma_2 ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_2 ~ normal(0, pi_u_mu_2);
  u_beta_2 ~ normal(0, pi_u_beta_2);
  u_alpha_2 ~ normal(0, pi_u_alpha_2);

  s_mu_2 ~ cauchy(0, pi_s_mu_2);
  s_beta_2 ~ cauchy(0, pi_s_beta_2);
  s_alpha_2 ~ cauchy(0, pi_s_alpha_2);
  s_sigma_2 ~ cauchy(0, pi_s_sigma_2);

  // Specifying priors for the parameters:
  mu_2 ~ normal(u_mu_2, s_mu_2);
  beta_2 ~ normal(u_beta_2[index_x_2], s_beta_2[index_x_2]);
  alpha_2 ~ normal(u_alpha_2, s_alpha_2);
  sigma_2 ~ cauchy(0, s_sigma_2);

  // Specifying the likelihood:
  y_2 ~ normal(expectation_2, sigma_vec_2);

  // Generating data from the model:
  y_gen_2 ~ normal(expectation_2, sigma_vec_2);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_2_3 ~ cauchy(0, phi);
  pi_s_eta_2_3 ~ cauchy(0, phi);
  s_eta_2_3 ~ cauchy(0, pi_s_eta_2_3);
  u_eta_2_3 ~ normal(0, pi_u_eta_2_3);
  eta_2_3 ~ normal(u_eta_2_3, s_eta_2_3);

  //// Third response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_3 ~ cauchy(0,  phi);
  pi_u_beta_3 ~ cauchy(0,  phi);
  pi_u_alpha_3 ~ cauchy(0,  phi);

  pi_s_mu_3 ~ cauchy(0,  phi);
  pi_s_beta_3 ~ cauchy(0,  phi);
  pi_s_alpha_3 ~ cauchy(0,  phi);
  pi_s_sigma_3 ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_3 ~ normal(0, pi_u_mu_3);
  u_beta_3 ~ normal(0, pi_u_beta_3);
  u_alpha_3 ~ normal(0, pi_u_alpha_3);

  s_mu_3 ~ cauchy(0, pi_s_mu_3);
  s_beta_3 ~ cauchy(0, pi_s_beta_3);
  s_alpha_3 ~ cauchy(0, pi_s_alpha_3);
  s_sigma_3 ~ cauchy(0, pi_s_sigma_3);

  // Specifying priors for the parameters:
  mu_3 ~ normal(u_mu_3, s_mu_3);
  beta_3 ~ normal(u_beta_3[index_x_3], s_beta_3[index_x_3]);
  alpha_3 ~ normal(u_alpha_3, s_alpha_3);
  sigma_3 ~ cauchy(0, s_sigma_3);

  // Specifying the likelihood:
  y_3 ~ normal(expectation_3, sigma_vec_3);

  // Generating data from the model:
  y_gen_3 ~ normal(expectation_3, sigma_vec_3);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_3_4 ~ cauchy(0, phi);
  pi_s_eta_3_4 ~ cauchy(0, phi);
  s_eta_3_4 ~ cauchy(0, pi_s_eta_3_4);
  u_eta_3_4 ~ normal(0, pi_u_eta_3_4);
  eta_3_4 ~ normal(u_eta_3_4, s_eta_3_4);

  //// Fourth response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_4 ~ cauchy(0,  phi);
  pi_u_beta_4 ~ cauchy(0,  phi);
  pi_u_alpha_4 ~ cauchy(0,  phi);

  pi_s_mu_4 ~ cauchy(0,  phi);
  pi_s_beta_4 ~ cauchy(0,  phi);
  pi_s_alpha_4 ~ cauchy(0,  phi);
  pi_s_sigma_4 ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_4 ~ normal(0, pi_u_mu_4);
  u_beta_4 ~ normal(0, pi_u_beta_4);
  u_alpha_4 ~ normal(0, pi_u_alpha_4);

  s_mu_4 ~ cauchy(0, pi_s_mu_4);
  s_beta_4 ~ cauchy(0, pi_s_beta_4);
  s_alpha_4 ~ cauchy(0, pi_s_alpha_4);
  s_sigma_4 ~ cauchy(0, pi_s_sigma_4);

  // Specifying priors for the parameters:
  mu_4 ~ normal(u_mu_4, s_mu_4);
  beta_4 ~ normal(u_beta_4[index_x_4], s_beta_4[index_x_4]);
  alpha_4 ~ normal(u_alpha_4, s_alpha_4);
  sigma_4 ~ cauchy(0, s_sigma_4);

  // Specifying the likelihood:
  y_4 ~ normal(expectation_4, sigma_vec_4);

  // Generating data from the model:
  y_gen_4 ~ normal(expectation_4, sigma_vec_4);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_4_5 ~ cauchy(0, phi);
  pi_s_eta_4_5 ~ cauchy(0, phi);
  s_eta_4_5 ~ cauchy(0, pi_s_eta_4_5);
  u_eta_4_5 ~ normal(0, pi_u_eta_4_5);
  eta_4_5 ~ normal(u_eta_4_5, s_eta_4_5);

  //// Fifth response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_5 ~ cauchy(0,  phi);
  pi_u_beta_5 ~ cauchy(0,  phi);
  pi_u_alpha_5 ~ cauchy(0,  phi);

  pi_s_mu_5 ~ cauchy(0,  phi);
  pi_s_beta_5 ~ cauchy(0,  phi);
  pi_s_alpha_5 ~ cauchy(0,  phi);
  pi_s_sigma_5 ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_5 ~ normal(0, pi_u_mu_5);
  u_beta_5 ~ normal(0, pi_u_beta_5);
  u_alpha_5 ~ normal(0, pi_u_alpha_5);

  s_mu_5 ~ cauchy(0, pi_s_mu_5);
  s_beta_5 ~ cauchy(0, pi_s_beta_5);
  s_alpha_5 ~ cauchy(0, pi_s_alpha_5);
  s_sigma_5 ~ cauchy(0, pi_s_sigma_5);

  // Specifying priors for the parameters:
  mu_5 ~ normal(u_mu_5, s_mu_5);
  beta_5 ~ normal(u_beta_5[index_x_5], s_beta_5[index_x_5]);
  alpha_5 ~ normal(u_alpha_5, s_alpha_5);
  sigma_5 ~ cauchy(0, s_sigma_5);

  // Specifying the likelihood:
  y_5 ~ normal(expectation_5, sigma_vec_5);

  // Generating data from the model:
  y_gen_5 ~ normal(expectation_5, sigma_vec_5);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_u_eta_5_6 ~ cauchy(0, phi);
  pi_s_eta_5_6 ~ cauchy(0, phi);
  s_eta_5_6 ~ cauchy(0, pi_s_eta_5_6);
  u_eta_5_6 ~ normal(0, pi_u_eta_5_6);
  eta_5_6 ~ normal(u_eta_5_6, s_eta_5_6);

  //// Sixth response variable conditionals probability distributions:

  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_6 ~ cauchy(0,  phi);
  pi_u_beta_6 ~ cauchy(0,  phi);
  pi_u_alpha_6 ~ cauchy(0,  phi);

  pi_s_mu_6 ~ cauchy(0,  phi);
  pi_s_beta_6 ~ cauchy(0,  phi);
  pi_s_alpha_6 ~ cauchy(0,  phi);
  pi_s_sigma_6 ~ cauchy(0,  phi);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_6 ~ normal(0, pi_u_mu_6);
  u_beta_6 ~ normal(0, pi_u_beta_6);
  u_alpha_6 ~ normal(0, pi_u_alpha_6);

  s_mu_6 ~ cauchy(0, pi_s_mu_6);
  s_beta_6 ~ cauchy(0, pi_s_beta_6);
  s_alpha_6 ~ cauchy(0, pi_s_alpha_6);
  s_sigma_6 ~ cauchy(0, pi_s_sigma_6);

  // Specifying priors for the parameters:
  mu_6 ~ normal(u_mu_6, s_mu_6);
  beta_6 ~ normal(u_beta_6[index_x_6], s_beta_6[index_x_6]);
  alpha_6 ~ normal(u_alpha_6, s_alpha_6);
  sigma_6 ~ cauchy(0, s_sigma_6);

  // Specifying the likelihood:
  y_6 ~ normal(expectation_6, sigma_vec_6);

  // Generating data from the model:
  y_gen_6 ~ normal(expectation_6, sigma_vec_6);

}