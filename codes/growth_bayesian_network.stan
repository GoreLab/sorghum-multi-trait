
functions {

  // User-defined vectorized logistic growth function:
  vector logistic_growth(real a, real c, vector r, real t) {

      return (c ./ (1 + a * exp(-r * t)));

  }

}

data {

  // Number of row entries of the matrices or vectors:
  int<lower=1> n;
  int<lower=1> n_t;  
  int<lower=1> p_z;

  // Feature matrix:
  matrix[n, p_z] Z;
  
  // Phenotypic vectors:
  vector[n] y[n_t];

  // Time points:
  vector[n_t] time_points;

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

  // Logistic growth function non-bin parameter/hyperparameters:
  real<lower=0> pi_s_a;
  real<lower=0> s_a;
  real<lower=0> a;
  real<lower=0> pi_s_c;
  real<lower=0> s_c;
  real<lower=0> c;

  // Residual parameter/hyperparameters:
  real<lower=0> pi_s_sigma;
  real<lower=0> s_sigma;
  vector<lower=0>[n_t] sigma;

  // Defining variable to generate data from the model:
  vector[n] y_gen[n_t];

}

transformed parameters {

  // Declaring variables to receive input:
  vector[n] expectation[n_t];
  vector<lower=0>[n] r;
  
  // Compute genotypic values:
  r = mu + Z * alpha;

  // Computing the expectation of the likelihood function:
  for (t in 1:n_t)
    expectation[t] = logistic_growth(a, c, r, time_points[t]);
 
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

  //// Conditional probabilities distributions for logistic growth function parameters:
  pi_s_a ~ cauchy(0,  phi);
  s_a ~ cauchy(0, pi_s_a);
  a ~ normal(0, s_a);

  pi_s_c ~ cauchy(0,  phi);
  s_c ~ cauchy(0, pi_s_c);
  c ~ normal(0, s_c);

  //// Conditional probabilities distributions for residuals:
  pi_s_sigma ~ cauchy(0,  phi);
  s_sigma ~ cauchy(0, pi_s_sigma);
  sigma ~ cauchy(0, s_sigma);

  for (t in 1:n_t) {

    // Specifying the likelihood:
    y[t] ~ normal(expectation[t], sigma[t]);

    // Generating data from the model:
    y_gen[t] ~ normal(expectation[t], sigma[t]);

  }

}
