
  // Parameters:
  real mu_0;
  vector[p_x] beta_0;
  vector<lower=0>[p_r] sigma_0;

  // First level hyperparameters:
  real u_mu_0;
  vector[p_i] u_beta_0;

  real<lower=0> s_mu_0;
  vector<lower=0>[p_i] s_beta_0;
  real<lower=0> s_sigma_0;

  // Second level hyperparameters:
  real<lower=0> pi_u_mu_0;
  real<lower=0> pi_u_beta_0;

  real<lower=0> pi_s_mu_0;
  real<lower=0> pi_s_beta_0;
  real<lower=0> pi_s_sigma_0;

  // Defining variable to generate data from the model:
  real y_gen_0[n_x];
