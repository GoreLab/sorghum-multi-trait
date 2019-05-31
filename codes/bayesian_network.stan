
data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n;
  int<lower=1> p_z;

  // Feature matrices:
  matrix[n, p_z] Z;
  
  // Phenotypic vector:
  real y[n];

  // Global known hyperparameter:
  real phi;

}

parameters {

  // Population mean parameter/hyperparameters:
  real<lower=0> pi_s_mu;
  real<lower=0> s_mu;
  real mu;

  // Features parameter/hyperparameters:
  real<lower=0> pi_s_alpha;
  real<lower=0> s_alpha;
  vector[p_z] alpha;  

  // Residual parameter/hyperparameters:
  real<lower=0> pi_s_sigma;
  real<lower=0> s_sigma;
  real<lower=0> sigma;

  // Defining variable to generate data from the model:
  real y_gen[n];

}

transformed parameters {

  // Declaring variables to receive input:
  vector[n] expectation;

  // Computing the expectation of the likelihood function:
  expectation = mu + Z * alpha;

}

model {

  //// Conditional probabilities distributions for mean:
  pi_s_mu ~ cauchy(0,  phi);
  s_mu ~ cauchy(0, pi_s_mu);
  mu ~ normal(0, s_mu);

  //// Conditional probabilities distributions for features:
  pi_s_alpha ~ cauchy(0,  phi);
  s_alpha ~ cauchy(0, pi_s_alpha);
  alpha ~ normal(0, s_alpha);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma ~ cauchy(0,  phi);
  s_sigma ~ cauchy(0, pi_s_sigma);
  sigma ~ cauchy(0, s_sigma);

  // Specifying the likelihood:
  y ~ normal(expectation, sigma);

  // Generating data from the model:
  y_gen ~ normal(expectation, sigma);

}
