data {
  int<lower=0> N;         // number of data points
  int<lower=1> D;         // dimensionality of each data point
  matrix[N, D] y;         // data matrix (each row is a data point)
}

parameters {
  simplex[3] lambda;        // mixture weights
  matrix[3, D] mu;          // mean vectors for each cluster
  array[3] cov_matrix[D] Sigma;
}

model {
  array[3] real ps;

  vector[D] group;

  group[1] = 91.184211; group[2] = 349.973684; //group[3] = 172.644737; group[4] = 114.0;
  mu[1] ~ normal(group, 10);

  group[1] = 99.305556; group[2] = 493.944444; //group[3] = 288.000000; group[4] = 208.972222;
  mu[2] ~ normal(group, 10);

  group[1] = 217.666667; group[2] = 1043.757576; //group[3] = 106.000000; group[4] = 318.878788;
  mu[3] ~ normal(group, 10);


  // Prior
  vector[3] alpha;
  alpha[1] = 76.0;
  alpha[2] = 36.0;
  alpha[3] = 33.0;
  lambda ~ dirichlet(alpha);
  
  for (k in 1:3) {
    Sigma[k] ~ inv_wishart(D + 1, diag_matrix(rep_vector(1, D)));  
  }

  for (n in 1:N) {
    for (k in 1:3) {
      ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}

generated quantities {
  matrix[N, 3] label_prob;   // probabilities of each row's label
  
  for (n in 1:N) {
    array[3] real log_ps;
    real max_log_ps;
    
    for (k in 1:3) {
      log_ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]); 
    }
    
    max_log_ps = max(log_ps);
    
    for (k in 1:3) {
      label_prob[n, k] = exp(log_ps[k] - max_log_ps);
    }
    
    // Normalize to get probabilities
    label_prob[n] /= sum(label_prob[n]);

  }
}

