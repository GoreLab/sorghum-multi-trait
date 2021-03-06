
data {

  // Number of features:
  int<lower=1> p_z;

  // Number of residual groups:
  int<lower=1> p_res;

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_0;

  // Feature matrix:
  matrix[n_0, p_z] Z_0;
  
  // Phenotypic vector:
  real y_0[n_0];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_1;

  // Feature matrix:
  matrix[n_1, p_z] Z_1;

  // Phenotypic vector:
  real y_1[n_1];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_2;

  // Feature matrix:
  matrix[n_2, p_z] Z_2;

  // Phenotypic vector:
  real y_2[n_2];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_3;

  // Feature matrix:
  matrix[n_3, p_z] Z_3;

  // Phenotypic vector:
  real y_3[n_3];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_4;

  // Feature matrix:
  matrix[n_4, p_z] Z_4;

  // Phenotypic vector:
  real y_4[n_4];

  // Number of row entries of the matrices or vectors:
  int<lower=1> n_5;

  // Feature matrix:
  matrix[n_5, p_z] Z_5;

  // Phenotypic vector:
  real y_5[n_5];

  // Global hyperparameter:
  real phi;

}

parameters {

  // Heterogeneous residuals parameter/hyperparameters:
  real<lower=0> pi_s_sigma;
  real<lower=0> s_sigma;
  vector<lower=0>[p_res] sigma;

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_0;
  real<lower=0> s_mu_0;
  real mu_0;

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

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_1;
  real<lower=0> s_mu_1;
  real mu_1;

  // Defining variable to generate data from the model:
  real y_gen_1[n_1];

  // Temporal Genetic effect parameter/hyperparameters:
  real<lower=0> pi_s_eta_1_2;
  real<lower=0> s_eta_1_2;
  vector[p_z] eta_1_2;

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_2;
  real<lower=0> s_mu_2;
  real mu_2;

  // Defining variable to generate data from the model:
  real y_gen_2[n_2];

  // Temporal Genetic effect parameter/hyperparameters:
  real<lower=0> pi_s_eta_2_3;
  real<lower=0> s_eta_2_3;
  vector[p_z] eta_2_3;

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_3;
  real<lower=0> s_mu_3;
  real mu_3;

  // Defining variable to generate data from the model:
  real y_gen_3[n_3];

  // Temporal Genetic effect parameter/hyperparameters:
  real<lower=0> pi_s_eta_3_4;
  real<lower=0> s_eta_3_4;
  vector[p_z] eta_3_4;

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_4;
  real<lower=0> s_mu_4;
  real mu_4;

  // Defining variable to generate data from the model:
  real y_gen_4[n_4];

  // Temporal Genetic effect parameter/hyperparameters:
  real<lower=0> pi_s_eta_4_5;
  real<lower=0> s_eta_4_5;
  vector[p_z] eta_4_5;

  // Population mean effect parameter/hyperparameters:
  real<lower=0> pi_s_mu_5;
  real<lower=0> s_mu_5;
  real mu_5;

  // Defining variable to generate data from the model:
  real y_gen_5[n_5];

}

transformed parameters {

  // Declaring variables to receive input:
  vector[p_z] alpha_1;
  vector[p_z] alpha_2;
  vector[p_z] alpha_3;
  vector[p_z] alpha_4;
  vector[p_z] alpha_5;
  vector[n_0] expectation_0;
  vector[n_1] expectation_1;
  vector[n_2] expectation_2;
  vector[n_3] expectation_3;
  vector[n_4] expectation_4;
  vector[n_5] expectation_5;

  // Computing the expectation of the likelihood function:
  expectation_0 = mu_0 + Z_0 * alpha_0;

  // Computing the global genetic effect:
  alpha_1 = (alpha_0 + eta_0_1);

  // Computing the expectation of the likelihood function:
  expectation_1 = mu_1 + Z_1 * alpha_1;

  // Computing the global genetic effect:
  alpha_2 = (alpha_1 + eta_1_2);

  // Computing the expectation of the likelihood function:
  expectation_2 = mu_2 + Z_2 * alpha_2;

  // Computing the global genetic effect:
  alpha_3 = (alpha_2 + eta_2_3);

  // Computing the expectation of the likelihood function:
  expectation_3 = mu_3 + Z_3 * alpha_3;

  // Computing the global genetic effect:
  alpha_4 = (alpha_3 + eta_3_4);

  // Computing the expectation of the likelihood function:
  expectation_4 = mu_4 + Z_4 * alpha_4;

  // Computing the global genetic effect:
  alpha_5 = (alpha_4 + eta_4_5);

  // Computing the expectation of the likelihood function:
  expectation_5 = mu_5 + Z_5 * alpha_5;

}

model {

  //// First response variable conditionals probability distributions:

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma ~ cauchy(0, phi);
  s_sigma ~ cauchy(0, pi_s_sigma);
  sigma ~ normal(0, s_sigma);

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_0 ~ cauchy(0, phi);
  s_mu_0 ~ cauchy(0, pi_s_mu_0);
  mu_0 ~ normal(0, s_mu_0);

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

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_1 ~ cauchy(0, phi);
  s_mu_1 ~ cauchy(0, pi_s_mu_1);
  mu_1 ~ normal(0, s_mu_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma[2]);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma[2]);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_s_eta_1_2 ~ cauchy(0, phi);
  s_eta_1_2 ~ cauchy(0, pi_s_eta_1_2);
  eta_1_2 ~ normal(0, s_eta_1_2);

  //// Third response variable conditionals probability distributions:

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_2 ~ cauchy(0, phi);
  s_mu_2 ~ cauchy(0, pi_s_mu_2);
  mu_2 ~ normal(0, s_mu_2);

  // Specifying the likelihood:
  y_2 ~ normal(expectation_2, sigma[3]);

  // Generating data from the model:
  y_gen_2 ~ normal(expectation_2, sigma[3]);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_s_eta_2_3 ~ cauchy(0, phi);
  s_eta_2_3 ~ cauchy(0, pi_s_eta_2_3);
  eta_2_3 ~ normal(0, s_eta_2_3);

  //// Third response variable conditionals probability distributions:

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_3 ~ cauchy(0, phi);
  s_mu_3 ~ cauchy(0, pi_s_mu_3);
  mu_3 ~ normal(0, s_mu_3);

  // Specifying the likelihood:
  y_3 ~ normal(expectation_3, sigma[4]);

  // Generating data from the model:
  y_gen_3 ~ normal(expectation_3, sigma[4]);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_s_eta_3_4 ~ cauchy(0, phi);
  s_eta_3_4 ~ cauchy(0, pi_s_eta_3_4);
  eta_3_4 ~ normal(0, s_eta_3_4);

  //// Fourth response variable conditionals probability distributions:

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_4 ~ cauchy(0, phi);
  s_mu_4 ~ cauchy(0, pi_s_mu_4);
  mu_4 ~ normal(0, s_mu_4);

  // Specifying the likelihood:
  y_4 ~ normal(expectation_4, sigma[5]);

  // Generating data from the model:
  y_gen_4 ~ normal(expectation_4, sigma[5]);

  //// Conditional probabilities distributions that creates dependecy between the responses:
  pi_s_eta_4_5 ~ cauchy(0, phi);
  s_eta_4_5 ~ cauchy(0, pi_s_eta_4_5);
  eta_4_5 ~ normal(0, s_eta_4_5);

  //// Fifth response variable conditionals probability distributions:

  //// Conditional probabilities distributions for population mean:
  pi_s_mu_5 ~ cauchy(0, phi);
  s_mu_5 ~ cauchy(0, pi_s_mu_5);
  mu_5 ~ normal(0, s_mu_5);

  // Specifying the likelihood:
  y_5 ~ normal(expectation_5, sigma[6]);

  // Generating data from the model:
  y_gen_5 ~ normal(expectation_5, sigma[6]);

}
