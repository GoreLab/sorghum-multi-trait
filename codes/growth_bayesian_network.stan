
functions {

  # User-defined vectorized logistic growth function:
  vector logistic_growth(real a, real c, real r, vector t) {

      return (c ./ (1 + a .* exp(-r * t)));

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
  vector[n_t] y_0[n];

  // Time points:
  real time_points[n_t];

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
  vector[n] a;
  real<lower=0> pi_s_c;
  real<lower=0> s_c;
  vector[n] c;

  // Residual parameter/hyperparameters:
  real<lower=0> pi_s_sigma;
  real<lower=0> s_sigma;
  vector<lower=0>[n_t] sigma;

  // Defining variable to generate data from the model:
  vector[n_t] y_gen[n];

}

transformed parameters {

  // Declaring variables to receive input:
  vector[n_t] expectation[n];
  vector[n] r;

  // Compute genotypic values:
  r = mu + Z * alpha;

  // Computing the expectation of the likelihood function:
  for (i in 1:n)
    expectation[i] = logistic_growth(a[i], c[i], r[i], time_points);
 
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

  for (i in 1:n) {

    // Specifying the likelihood:
    y[i] ~ normal(expectation[i], sigma);

    // Generating data from the model:
    y_gen[i] ~ normal(expectation[i], sigma);

  }

}
