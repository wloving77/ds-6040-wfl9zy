#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# 
# 
# ## Instructions
# 
# Please complete this Jupyter notebook and then convert it to a `.py` file called `assignment4.py`. Upload this file to Gradescope, and await feedback. 
# 
# For manual grading, you will need to upload additional files to separate Gradescope portals.
# 
# You may submit as many times as you want up until the deadline. Only your latest submission counts toward your grade.
# 
# Some tests are hidden and some are visible. The outcome of the visible checks will be displayed to you immediately after you submit to Gradescope. The hidden test outcomes will be revealed after final scores are published. 
# 
# This means that an important part of any strategy is to **start early** and **lock in all the visible test points**. After that, brainstorm what the hidden checks could be and collaborate with your teammates.
# 

# ### Problem 1: Dirichlet-Multinomial
# 
# If $y = (y_1, \ldots, y_k)$ is a length $k$ count vector (e.g. $(1,3,4,0,2)$ and $k=5$), we could let it have a multinomial distribution. 
# $$
# L(y \mid \theta) \propto \prod_{i=1}^k \theta_i^{y_i}
# $$
# 
# We could also put a $\text{Dirichlet}(\alpha_1, \ldots, \alpha_k)$ prior on the unknown $\theta$:
# $$
# \pi(\theta) = \text{Dirichlet}(\alpha_1, \ldots, \alpha_k) \propto \prod_{i=1}^k \theta_i^{\alpha_i - 1}.
# $$
# 
# In lecture we showed that the resulting posterior is 
# $$
# \theta \mid y \sim \text{Dirichlet}(\alpha_1 + y_1, \ldots, \alpha_k + y_k)
# $$
# 
# Now suppose that we have the same prior, but we have $N > 1$ observed vectors. Derive the posterior in this situation. Show all your work!
# 
# Hint: it is customary to let the row index correspond to the observation number; i.e.
# 
# $$
# y = 
# \begin{bmatrix}
# y_1 \\
# y_2 \\
# \vdots \\
# y_N
# \end{bmatrix}
# =
# \begin{bmatrix}
# y_{1,1} & y_{1,2} & \cdots & y_{1,k} \\
# \vdots  & \vdots & \ddots & \vdots \\
# y_{N,1} & y_{N,2} & \cdots & y_{N,k} \\
# \end{bmatrix}
# $$
# 
# Hint 2: we assume each row/observation is independent and identically distributed, so
# 
# $$
# L(y_1, \ldots, y_N \mid \theta) = \prod_{r=1}^N L(y_r \mid \theta) 
# $$

# ------------------
# # Answer:
# 
# The likelihood for each individual \( y_r \) is:
# $$
# L(y_r \mid \theta) \propto \prod_{i=1}^k \theta_i^{y_{r,i}}
# $$
# 
# Thus, the joint likelihood is:
# $$
# L(y_1, \ldots, y_N \mid \theta) \propto \prod_{r=1}^N \prod_{i=1}^k \theta_i^{y_{r,i}} = \prod_{i=1}^k \theta_i^{\sum_{r=1}^N y_{r,i}}
# $$
# 
# Combining this with the Dirichlet prior:
# $$
# \pi(\theta) \propto \prod_{i=1}^k \theta_i^{\alpha_i - 1}
# $$
# 
# The posterior is proportional to the product of the likelihood and the prior:
# $$
# \pi(\theta \mid y_1, \ldots, y_N) \propto L(y_1, \ldots, y_N \mid \theta) \pi(\theta)
# $$
# 
# Substituting the likelihood and prior:
# $$
# \pi(\theta \mid y_1, \ldots, y_N) \propto \left( \prod_{i=1}^k \theta_i^{\sum_{r=1}^N y_{r,i}} \right) \left( \prod_{i=1}^k \theta_i^{\alpha_i - 1} \right)
# $$
# 
# Combining the exponents:
# $$
# \pi(\theta \mid y_1, \ldots, y_N) \propto \prod_{i=1}^k \theta_i^{\alpha_i - 1 + \sum_{r=1}^N y_{r,i}}
# $$
# 
# This is the kernel of a Dirichlet distribution with updated parameters:
# $$
# \theta \mid y_1, \ldots, y_N \sim \text{Dirichlet} \left( \alpha_1 + \sum_{r=1}^N y_{r,1}, \ldots, \alpha_k + \sum_{r=1}^N y_{r,k} \right)
# $$
# 
# ------------------

# ### Problem 2: Roulette!
# 
# 
# 
# ![roulette.jpg](attachment:f7e1207c-2e7d-4712-9d1a-2306af682ee3.jpg)
# 
# 
# 
# 

# In the game of (American) Roulette, you guess where the ball will land after the wheel is spun. If all spaces are equally-likely, the house has an advantage no matter what you do. However, if you can spot a "bias" in the wheel, you might have an advantage!
# 
# In this example, the parameter $\theta = (\theta_1, \ldots, \theta_{36}, \theta_{37}, \theta_{38})$ represents the probabilities of each of the 38 possible ball-landing locations. $\theta_{37}$, $\theta_{38}$ will represent the "0" and "00" outcomes, respectively.
# 
# Suppose you have a $\text{Dirichlet}(\alpha_1, \ldots, \alpha_{38})$ prior:
# $$
# \pi(\theta) \propto \prod_{j=1}^{38} \theta_j^{\alpha_j-1}.
# $$
# 
# And you have $N$ vector-valued observations--one vector of counts for each day. We can use a **multinomial** likelihood for this data:
# $$
# L(y \mid \theta) = \prod_{i=1}^{100} L(y_i \mid \theta),
# $$
# 
# where
# $$
# L(y_i \mid \theta) \propto \prod_{j=1}^{38} \theta_j^{y_{ij}}
# $$

# 1.
# 
# Pick a prior by specifying the hyperparameters $\alpha_1, \ldots, \alpha_{38}$. Assign a Numpy array to the variable named `prior_hyperparams`

# In[29]:


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

prior_hyperparams = np.array([1] * 38)
prior_hyperparams


# 2. 
# 
# Consider the following (fake) data set `my_data` below. Construct the observation vectors. This data array should have day on the rows, and location on the columns. As a result, your final answer should have shape `(100,38)`
# 
# For example, the day three row vector consists of the 38 count integers:
# 
# $$
# y_3 = \begin{bmatrix}\text{(num 1s on day 3)}, & \cdots ,& \text{(num 38s on day 3)} \end{bmatrix}
# $$
# 
# Assign your answer to the numpy array `y_data`.
# 
# Hint: `.pivot_table()` is nice for this.

# In[10]:


# Define the possible numbers on the roulette wheel
numbers = np.arange(0, 38)  
colors = ['green'] + ['red', 'black'] * 18  + ['green']

# simulate some data
num_days = 100
num_spins_per_day = 100
num_rows = num_days*num_spins_per_day
my_data = pd.DataFrame({'number':np.random.choice(numbers, num_rows)})
my_data['color'] = my_data.number.apply( lambda num : colors[num])
my_data['day'] = np.repeat(np.arange(1,(num_spins_per_day+1)),num_days)
my_data


# In[19]:


y_data = my_data.pivot_table(index='day', columns='number', aggfunc='size', fill_value=0).values
y_data


# 3.
# 
# Calculate the Dirichlet posterior hyperparameters. Assign a 1-d Numpy array to the variable named `post_hyperparams`

# In[22]:


post_hyperparams = prior_hyperparams + y_data.sum(axis=0)
post_hyperparams


# 4.
# 
# If you can only bet one one number, which number are you picking? Possible answer choices include $1, \ldots, 36,37,38$ where the last two numbers follow the aforementioned convention. 
# 
# Assign your answer to `best_bet`
# 
# Hint: be careful...Python uses 0-based indexing but Roulette starts counting at $1$.
# 

# In[25]:


best_bet = np.argmax(post_hyperparams) + 1
best_bet


# 5.
# 
# 
# If the true model is that every outcome is equally likely, will you make money in the long run?
# 
# In other words, suppose that your prior is **super informative,** and simulate from the prior predictive distribution. For each spin of the wheel, calculate your profit (or loss). Use fixed bet sizes.
# 
# Visualize your simulated profit and losses appropriately, and justify your answer with these visualizations.

# In[40]:


# Prior Predictive Distribution, does not consider data

num_spins = 1000

theta = np.random.dirichlet(prior_hyperparams)
spins = np.random.choice(38, size=num_spins, p=theta)

bet_on_number = 1
bet_size = 1
payout = 35  # Payout for a single number bet in American Roulette

profits = []
cumulative_profit = 0

for spin in spins:
    if spin == bet_on_number:
        profit = bet_size * payout
    else:
        profit = -bet_size
    cumulative_profit += profit
    profits.append(cumulative_profit)

# Step 4: Visualize the results
# plt.figure(figsize=(12, 6))
# plt.plot(profits, label="Cumulative Profit")
# plt.axhline(0, color='red', linestyle='--', label="Break-even")
# plt.title("Simulated Cumulative Profit from Betting on a Single Number in Roulette")
# plt.xlabel("Number of Spins")
# plt.ylabel("Cumulative Profit")
# plt.legend()
# plt.show()


# ### Problem 3: Stocks
# 
# In example 2 of module 4, we showed that if one has multivariate normal observations, then the **Normal Inverse Wishart** prior is the conjugate prior.
# 
# In this example, we'll be using stocks data again, but with a **daily sampling frequency.**

# 1.
# 
# Download the data `stocks.csv` and assign it to a `pandas` `DataFrame` called `adj_prices`. Be sure to set the date as the index.
# 
# Calculate percent returns (scaled by $100$) and call the resulting `DataFrame` `rets`. After understanding where they come from, be sure to remove any `NaN`s.

# In[48]:


adj_prices = pd.read_csv("stocks.csv", index_col="Date")
rets = adj_prices.pct_change().dropna() * 100


# 2.
# 
# Write a function called `sim_data` that can simulate from either the prior predictive or the posterior predictive distribution.
# 
# Here's a template based on how I did it. You may use all of it or just some of it. However, do be sure to keep the same function signature :) Every input or output is either a `float` or a 2-d Numpy array.

# In[99]:


from scipy.stats import invwishart
from scipy.stats import multivariate_normal

def sim_data(nu0, Lambda0, mu0, kappa0, num_sims):
    sigma_samples = invwishart.rvs(df=nu0, scale=Lambda0, size=num_sims)

    # Ensure sigma_samples is always 3-dimensional
    if num_sims == 1:
        sigma_samples = np.array([sigma_samples])
    
    mu_samples = []
    for sigma in sigma_samples:
        mu = multivariate_normal.rvs(mean=mu0, cov=sigma / kappa0)
        mu_samples.append(mu)

    mu_samples = np.array(mu_samples)
    
    fake_y = []
    for i in range(num_sims):
        theta = multivariate_normal.rvs(mean=mu_samples[i], cov=sigma_samples[i])
        fake_y.append(theta)
    
    return np.array(fake_y)


# 3.
# 
# Pick a prior by assigning specific values to `nu0`, `Lambda0`, `mu0` and `kappa0`. Use your `sim_data()` function to choose wisely.

# In[156]:


nu0 = 5
Lambda0 = np.array([[1.0, 0.5], [0.5, 1.0]])
mu0 = np.zeros(2)
kappa0 = 1


# 4.
# 
# Calculate the posterior NIW distribution. Assign the distribution's hyperparameters to `nu_n`, `kappa_n`, `mu_n`, and `Lambda_n`

# In[164]:


X = rets.values

n = len(rets)

y_bar = np.mean(X, axis=0)
#S = np.sum([(x - X_bar).reshape(-1, 1) @ (x - X_bar).reshape(1, -1) for x in X], axis=0)
S = (rets.T @ rets).values

nu_n = nu0 + n
kappa_n = kappa0 + n
mu_n = (kappa0 * mu0 + n * y_bar) / (kappa0 + n)
Lambda_n = Lambda0 + S + (kappa0 * n / (kappa0 + n)) * np.outer(y_bar - mu0, y_bar - mu0)


# 5.
# 
# Simulate $1239$ observations from the posterior predictive distribution. You can use `sim_data()` for this, too. Assign your array to the variable `post_pred_sims`
# 
# 
# 
# Does this model fit lower-frequency stock returns better? Discuss.
# 
# 

# In[149]:


# uncomment after you have a working implementation of sim_data()!
post_pred_sims = sim_data(nu_n, Lambda_n, mu_n, kappa_n, 1239)
post_pred_sims[:5]


# 6. 
# 
# Now that you have a posterior for $\mu$ and $\Sigma$, that means you have an understanding of the distribution of future returns. Let's find the "optimal" portfolio weights! 
# 
# 
# Portfolio weights $w = w_1, \ldots, w_k$ are proportions of your wealth that you invest in each security. We want to find the vector that that **maximizes "risk-adjusted return"**
# 
# $$
# \underbrace{\mathbf{w}^\intercal \mathbb{E}_{\text{posterior}}[y]}_{\text{reward}} - \frac{1}{2} \gamma \underbrace{\mathbf{w}^\intercal \mathbb{V}_{\text{posterior}}[y] \mathbf{w}}_{\text{risk}}
# $$ 
# 
# subject to the constraint that you have a finite amount of money:
# 
# $$
# \sum_i w_i = 1
# $$
# 
# $\gamma > 0$ is a user-chosen risk-aversion parameter. Describe your choice of it, and run the function below to get your optimal portfolio weights. Discuss your results. Some examples of questions to consider are:
# 
#  - Do my weights indeed sum to $1$?
#  - What are some problems with choosing $\gamma$ very close to $0$?
#  - What are some problems with choosing $\gamma$ very large?
#  - What do I think most peoples' $\gamma$ are?
#  - Knowing what I know now, would I go back and change around the prior to get a posterior that gives me a better answer here? 

# In[92]:


def get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma, s = 1):
    k = len(mu_n)
    post_mean = mu_n
    post_var = Lambda_n / (nu_n - k - 1)*(1 + 1/kappa_n)
    V_inv = np.linalg.inv(post_var)
    ones = np.repeat(1, k)
    q1 = ones.transpose() @ V_inv @ post_mean
    q2 = ones.transpose() @ V_inv @ ones
    le_fraction = (q1 - gamma*s)/q2
    return V_inv @ (post_mean - le_fraction * ones) / gamma
    
#best_weights = get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma=???, s = 1)
best_weights = get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma=3, s = 1)


# In[93]:


best_weights


# In[ ]:




