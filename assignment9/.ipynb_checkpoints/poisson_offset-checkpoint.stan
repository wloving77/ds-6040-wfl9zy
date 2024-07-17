// Poisson Regression Model with Intercept in Stan

data {
  int<lower=0> N;           // Number of data points
  int<lower=0> K;           // Number of predictors
  array[N] int y;                 // Response variable (counts)
  matrix[N, K] X;           // Predictor matrix
  vector[N] modifier;
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
  y ~ poisson_log(modifier + alpha + X * beta);  // Vectorized Poisson regression with intercept
}

generated quantities {
  array[N] int y_pred;
  vector[N] lambda;
  
  for (n in 1:N) {
    lambda[n] = exp(modifier[n] + alpha + dot_product(X[n], beta));
    y_pred[n] = poisson_rng(lambda[n]);
  }
}