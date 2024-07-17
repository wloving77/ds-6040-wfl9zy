// Poisson Regression Model with Intercept in Stan

data {
  int<lower=0> N;           // Number of data points
  int<lower=0> K;           // Number of predictors
  array[N] int y;                 // Response variable (counts)
  matrix[N, K] X;           // Predictor matrix
}

parameters {
  real alpha;               // Intercept
  vector[K] beta;           // Coefficients for predictors
}

model {
  // Priors
  alpha ~ normal(0, 10);    // Weakly informative prior for intercept
  beta ~ normal(0, 10);     // Weakly informative prior for coefficients
  
  // Likelihood
  y ~ poisson_log(alpha + X * beta);  // Vectorized Poisson regression with intercept
}
