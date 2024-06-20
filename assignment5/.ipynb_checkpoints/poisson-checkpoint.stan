data {
  int<lower=0> n_trials;  
  array[n_trials] int<lower=0> y; 
}

parameters {
  real<lower=0> theta; // rate parameter
}

model {
  theta ~ lognormal(1, 0.5); // lognormal prior
  y ~ poisson(theta); // likelihood
}