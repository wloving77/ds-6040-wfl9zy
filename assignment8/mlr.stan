data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    vector[N] y;
}

parameters {
    real alpha; //intercept term
    vector[K] beta;
    real<lower=0> sigma;
}

model {
    alpha ~ normal(0,10); // intercept term
    beta ~ normal(0,10);
    sigma ~ lognormal(0,1);

    target += normal_lpdf(y | alpha + X * beta, sigma);

}

