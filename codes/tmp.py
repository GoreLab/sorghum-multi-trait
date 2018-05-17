
  // Specifying hyperpriors for the second level hyperparameters:
  pi_u_mu_1 ~ cauchy(0,  phi_1);
  pi_u_beta_1 ~ cauchy(0,  phi_1);

  pi_s_mu_1 ~ cauchy(0,  phi_1);
  pi_s_beta_1 ~ cauchy(0,  phi_1);
  pi_s_sigma_1 ~ cauchy(0,  phi_1);

  // Specifying hyperpriors for the first level hyperparameters:
  u_mu_1 ~ normal(0, pi_u_mu_1);
  u_beta_1 ~ normal(0, pi_u_beta_1);

  s_mu_1 ~ cauchy(0, pi_s_mu_1);
  s_beta_1 ~ cauchy(0, pi_s_beta_1);
  s_sigma_1 ~ cauchy(0, pi_s_sigma_1);

  // Specifying priors for the parameters:
  mu_1 ~ normal(u_mu_1, s_mu_1);
  beta_1 ~ normal(u_beta_1[index_x_1], s_beta_1[index_x_1]);
  sigma_1 ~ cauchy(0, s_sigma_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma_vec_1);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma_vec_1);


