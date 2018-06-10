  //// Conditional probabilities distributions for mean:
  pi_u_mu_1 ~ cauchy(0,  phi);
  pi_s_mu_1 ~ cauchy(0,  phi);
  u_mu_1 ~ normal(0, pi_u_mu_1);
  s_mu_1 ~ cauchy(0, pi_s_mu_1);
  mu_1 ~ normal(u_mu_1, s_mu_1);

  //// Conditional probabilities distributions for features:
  pi_u_alpha_1 ~ cauchy(0,  phi);
  pi_s_alpha_1 ~ cauchy(0,  phi);
  u_alpha_1 ~ normal(0, pi_u_alpha_1);
  s_alpha_1 ~ cauchy(0, pi_s_alpha_1);
  alpha_1 ~ normal(u_alpha_1, s_alpha_1);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma_1 ~ cauchy(0,  phi);
  s_sigma_1 ~ cauchy(0, pi_s_sigma_1);
  sigma_1 ~ cauchy(0, s_sigma_1);

  // Specifying the likelihood:
  y_1 ~ normal(expectation_1, sigma_1);

  // Generating data from the model:
  y_gen_1 ~ normal(expectation_1, sigma_1);
