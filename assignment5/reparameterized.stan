data {
  int<lower=0> n_trials;  
  array[n_trials] int<lower=0> y; 
}

parameters {
  real theta; // rate parameter
}

transformed parameters {
  real<lower=0> exp_theta;
  exp_theta = exp(theta);
}

model {
  theta ~ normal(0, 999); // lognormal prior
  y ~ poisson(theta); // likelihood
}