  //// Conditional probabilities distributions that creates dependecy between the responses across time:
  pi_u_d ~ cauchy(0, phi);
  pi_s_d ~ cauchy(0, phi);
  s_d ~ cauchy(0, pi_s_d);
  u_d ~ normal(0, pi_u_d);
  d ~ normal(u_d, s_d);
