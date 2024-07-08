data {
  int<lower=1> K;  // dimension of the normal distribution
  int<lower=1> N;  // number of observations
  matrix[N, K] y;  // observed data
  vector[K] mu_prior_mean;  // prior mean for mu
  real<lower=0> mu_prior_sd;  // prior standard deviation for mu
}

parameters {
  vector[K] mu;  // mean vector of the normal distribution
  cholesky_factor_corr[K] L_Omega;  // Cholesky factor of the correlation matrix
  vector<lower=0>[K] L_std;  // standard deviations
}

transformed parameters {
  matrix[K, K] L_Sigma;
  L_Sigma = diag_pre_multiply(L_std, L_Omega);
}

model {
  // Priors
  mu ~ normal(mu_prior_mean, mu_prior_sd);
  L_Omega ~ lkj_corr_cholesky(1);
  L_std ~ normal(0, 2.5);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ multi_normal_cholesky(mu, L_Sigma);
  }
}

generated quantities {
  matrix[N, K] y_sim;

  // Simulate from the posterior predictive distribution
  for (n in 1:N) {
    y_sim[n] = to_row_vector(multi_normal_cholesky_rng(mu, L_Sigma));
  }
}
