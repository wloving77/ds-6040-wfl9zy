data {
  int<lower=1> K;  // dimension of the normal distribution
  int<lower=1> N;
  matrix[N,K] y;
  vector[K] mu_0;  // prior mean vector
  real<lower=0> kappa_0;  // scaling factor for the normal distribution
  matrix[K, K] Lambda;  // scale matrix for the inverse Wishart distribution
  real<lower=K-1> nu;  // degrees of freedom for the inverse Wishart distribution
}

parameters {
  vector[K] mu;  // mean vector of the normal distribution
  cov_matrix[K] Sigma;  // covariance matrix
}

model {
  // Inverse Wishart prior for Sigma
  Sigma ~ inv_wishart(nu, Lambda);

  // Normal prior for mu conditional on Sigma
  mu ~ multi_normal(mu_0, Sigma / kappa_0);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ multi_normal(mu, Sigma);
  }
}