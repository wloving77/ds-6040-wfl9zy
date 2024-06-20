data {
  int<lower=0> m; // number of data points
  array[m] int<lower=0> n; // number of trials for each data point
  array[m] int<lower=0> y; // observed counts
}

parameters {
  real<lower=0, upper=1> theta; // constrained parameter
}

transformed parameters {
  real trans_theta;
  trans_theta = logit(theta);
}

model {
  theta ~ normal(0, 999); // normal prior for the unconstrained parameter
  for (i in 1:m)
    y[i] ~ binomial(n[i], inv_logit(trans_theta)); // binomial likelihood with inv_logit transformation
}
