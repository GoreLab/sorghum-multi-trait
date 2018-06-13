
data {

  // Number of features:
  int<lower=1> p_z;

  // Number of residual groups:
  int<lower=1> p_res;

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;

  // Feature vector/matrix:
  vector[n_0] X_0;
  matrix[n_0, p_z] Z_0;
  
  // Phenotypic vector:
  real y_0[n_0];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_1;

  // Feature vector/matrix:
  vector[n_1] X_1;
  matrix[n_1, p_z] Z_1;

  // Phenotypic vector:
  real y_1[n_1];

  // Global hyperparameter:
  real phi;

}

parameters {

  // Global dap effect parameter/hyperparameters:
  real<lower=0> pi_s_beta;
  real<lower=0> s_beta;
  real beta;

  // Heterogeneous residuals parameter/hyperparameters:
  real<lower=0> pi_s_sigma;
  real<lower=0> s_sigma;
  vector<lower=0>[p_res] sigma;

  // Feature parameter/hyperparameters:
  real<lower=0> pi_s_alpha_0;
  real<lower=0> s_alpha_0;
  vector[p_z] alpha_0;  

  // Defining variable to generate data from the model:
  real y_gen_0[n_0];

  // Temporal Genetic effect parameter/hyperparameters:
  real<lower=0> pi_s_eta_0_1;
  real<lower=0> s_eta_0_1;
  vector[p_z] eta_0_1;

  // Defining variable to generate data from the model:
  real y_gen_1[n_1];

}

transformed parameters {

  // Declaring variables to receive input:
  vector[p_z] alpha_1;
  vector[n_0] expectation_0;
  vector[n_1] expectation_1;

  // Computing the expectation of the likelihood function:
  expectation_0 = X_0 * beta + Z_0 * alpha_0;

  // Computing the global genetic effect:
  alpha_1 = (alpha_0 + eta_0_1);

  // Computing the expectation of the likelihood function:
  expectation_1 = X_1 * beta + Z_1 * alpha_1;

}

model {

  //// First response variable conditionals probability distributions:

  //// Conditional probabilities distributions that creates dependecy between the responses across time:
  pi_s_beta ~ cauchy(0, phi);
  s_beta ~ cauchy(0, pi_s_beta);
  beta ~ normal(0, s_beta);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma ~ cauchy(0, phi);
  s_sigma ~ cauchy(0, pi_s_sigma);
  sigma ~ normal(0, s_sigma);

  // Conditional probabilities distributions for features:
  pi_s_alpha_0 ~ cauchy(0,  phi);
  s_alpha_0 ~ cauchy(0, pi_s_alpha_0);
  alpha_0 ~ normal(0, s_alpha_0);

  // Specifying the likelihood:
  y_0 ~ normal(expectation_0, sigma[1]);

  // Generating data from the model:
  y_gen_0 ~ normal(expectation_0, sigma[1]);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_s_eta_0_1 ~ cauchy(0, phi);
  s_eta_0_1 ~ cauchy(0, pi_s_eta_0_1);
  eta_0_1 ~ normal(0, s_eta_0_1);

  //// Second response variable conditionals probability distributions:

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma[2]);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma[2]);

}
