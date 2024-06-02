#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# 
# 
# ## Instructions
# 
# Please complete this Jupyter notebook and then convert it to a `.py` file called `assignment2.py`. Upload this file to Gradescope, and await feedback. 
# 
# You may submit as many times as you want up until the deadline. Only your latest submission counts toward your grade.
# 
# Some tests are hidden and some are visible. The outcome of the visible checks will be displayed to you immediately after you submit to Gradescope. The hidden test outcomes will be revealed after final scores are published. 
# 
# This means that an important part of any strategy is to **start early** and **lock in all the visible test points**. After that, brainstorm what the hidden checks could be and collaborate with your teammates.
# 

# ### Problem 1
# 
# 
# Recall the derivation of the posterior when you had a binomial data point, and a uniform prior:
# 
# \begin{align}
# \pi(\theta \mid y) 
# &\propto L(y \mid \theta) \pi(\theta) \\
# &= \binom{n}{y} \theta^y (1-\theta)^{n-y} \mathbb{1}(0 < \theta < 1 ) \\
# &\propto
# \theta^y (1-\theta)^{n-y}
# \end{align}
# 
# Suppose $n=400$ and $y=250$.
# 
# 1.
# 
# What is the natural logarithm of the normalizing constant? In other words, what do we have to divide $\theta^y (1-\theta)^{n-y}$ by so that it integrates to $1$? Then take the natural log of that. 
# 
# Stated differently, what is $\log \int_0^1 \theta^y (1-\theta)^{n-y} \text{d} \theta$? 
# 
# 
# Assign your answer to `log_norm_const`
# 
# NB1: if we didn't use the logarithm, the normalizing constant would be *way* too close to $0$.
# 
# 
# NB2: You're not doing calculus here. The integral is just a special formula, and the formula is implemented in the `scipy.special` submodule.

# In[5]:


import scipy.special

alpha = 251
beta = 151

log_norm_const = scipy.special.betaln(alpha, beta)
log_norm_const


# 2. 
# 
# Are either of these dependent on the value of $\theta$? If yes, assign `True` to `dependent_on_theta`. Otherwise assign `False`

# In[3]:


# Theta is integrated out of the problem
dependent_on_theta = False


# ### Problem 2
# 
# 
# Suppose
# 
# $$\pi(\theta) = \text{Beta}(a,b),$$ 
#  
# and
# 
# $$L(y \mid \theta) = \text{Binomial}(n,\theta)$$
# 
# Show that $\pi(\theta \mid y) = \text{Beta}(a + y, n+b - y)$. Upload a scanned copy of your work to Gradescope portal.

# ----------------------
# 
# # Step 1: Prior Distribution
# $$\pi(\theta) = \text{Beta}(a, b) \propto \theta^{a-1} (1-\theta)^{b-1}$$
# 
# # Step 2: Likelihood Function
# $$L(y \mid \theta) = \text{Binomial}(n, \theta) \propto \theta^y (1-\theta)^{n-y}$$
# 
# # Step 3: Posterior Distribution using Bayes' Theorem
# $$\pi(\theta \mid y) \propto L(y \mid \theta) \pi(\theta)$$
# $$\pi(\theta \mid y) \propto \theta^y (1-\theta)^{n-y} \times \theta^{a-1} (1-\theta)^{b-1}$$
# 
# # Step 4: Combine Terms
# $$\pi(\theta \mid y) \propto \theta^{y + (a-1)} (1-\theta)^{(n-y) + (b-1)}$$
# 
# # Step 5: Simplify the Exponents
# $$\pi(\theta \mid y) \propto \theta^{(a+y-1)} (1-\theta)^{(n-y+b-1)}$$
# 
# # Step 6: Identify the Posterior Distribution
# $$\pi(\theta \mid y) = \text{Beta}(a + y, n + b - y)$$
# 
# --------------------

# ### Problem 3
# 
# 
# Suppose
# 
# $$\pi(\theta) = \text{Beta}(a,b),$$ 
#  
# for some fixed/chosen $a,b > 0$. Suppose further that you have $m > 1$ count data points $y_1, \ldots, y_m$, each having a $\text{Binomial}(n,\theta)$ distribution.
# 
# 1. What is the likelihood of $y_1, \ldots, y_m$ assuming they're all independent (conditioning on one $\theta$ value)?
# 2. What is the posterior distribution?
# 
# Upload a scanned copy of your work to Gradescope portal.

# -------------
# 
# # Step 1: Likelihood of y_1, ..., y_m
# $$L(y_i \mid \theta) \propto \theta^{y_i} (1-\theta)^{n-y_i}$$
# 
# # Joint Likelihood assuming independence
# $$L(y_1, \ldots, y_m \mid \theta) \propto \prod_{i=1}^m L(y_i \mid \theta) \propto \prod_{i=1}^m \left(\theta^{y_i} (1-\theta)^{n-y_i} \right)$$
# 
# # Simplifying the product of binomial coefficients, ANSWER:
# $$L(y_1, \ldots, y_m \mid \theta) \propto \left( \prod_{i=1}^m \right) \theta^{\sum_{i=1}^m y_i} (1-\theta)^{\sum_{i=1}^m (n-y_i)}$$
# 
# # Step 2: Posterior Distribution using Bayes' Theorem
# $$\pi(\theta \mid y_1, \ldots, y_m) \propto L(y_1, \ldots, y_m \mid \theta) \pi(\theta)$$
# 
# # Substitute prior and likelihood
# $$\pi(\theta \mid y_1, \ldots, y_m) \propto \theta^{\sum_{i=1}^m y_i} (1-\theta)^{mn - \sum_{i=1}^m y_i} \times \theta^{a-1} (1-\theta)^{b-1}$$
# 
# # Combine terms involving Theta
# $$\pi(\theta \mid y_1, \ldots, y_m) \propto \theta^{\sum_{i=1}^m y_i + a - 1} (1-\theta)^{mn - \sum_{i=1}^m y_i + b - 1}$$
# 
# # Identify the Posterior Distribution
# $$ \pi(\theta \mid y_1, \ldots, y_m) = \text{Beta}(a + \sum_{i=1}^m y_i, b + mn - \sum_{i=1}^m y_i)$$ 
# 
# ---------------

# ### Problem 4: Roulette!
# 
# In the game of **Roulette** the croupier spins a wheel and a ball, and you bet on where the ball will end up. Suppose you're interested in testing whether all possible outcomes are equally likely. Consider the fake data below.
# 
# ![roulette.jpg](roulette.jpg)

# In[7]:


# do not edit this cell!
import numpy as np
import pandas as pd

# Define the possible numbers on the roulette wheel
numbers = np.arange(0, 38)  # 0 to 36 for numbers, 37 for double zero
# Define the colors of the numbers
colors = ['green'] + ['red', 'black'] * 18  + ['green']

num_rows = 100
my_data = pd.DataFrame({'number':np.random.choice(numbers, num_rows)})
my_data['color'] = my_data.number.apply( lambda num : colors[num])
my_data.head()


# Suppose $\theta$ is the probability the ball lands on red.
# 
# 1. Choose a Beta prior for $\theta$. Assign your prior hyperparameters to the variables `prior_hyperparam1` and `prior_hyperparam2`. Make sure the mean of your prior is $18/38$!
# 
# Hint: do the previous problem first, and notice that, in this case, $n=1$ and $m=100$

# In[9]:


# a/a+b = 18/38
# simplify and end up with a/b = 9/10
# Mean of Beta = a/a+b, 9/9+10 = 9/18 = 18/38
prior_hyperparam1 = 9
prior_hyperparam2 = 10


# 2. Use the simulated data above, and come up with a posterior. Assign the parameters of the beta distribution to `posterior_hyperparam1` and `posterior_hyperparam2`
# 
# 

# In[14]:


num_red = my_data[my_data['color'] == 'red'].shape[0]
num_non_red = my_data[my_data['color'] != 'red'].shape[0]

#posterior_hyperparam1 = a + y
#posterior_hyperparam2 = b + (n-y)

posterior_hyperparam1 = prior_hyperparam1 + num_red
posterior_hyperparam2 = prior_hyperparam2 + num_non_red

posterior_hyperparam1, posterior_hyperparam2


# 3. Calculate a 95% *credible interval* for theta. Assign your answer to a `tuple` called `my_interval`
# 

# In[16]:


import scipy.stats as stats

lower_bound = stats.beta.ppf(0.025, posterior_hyperparam1, posterior_hyperparam2)
upper_bound = stats.beta.ppf(0.975, posterior_hyperparam1, posterior_hyperparam2)

my_interval = (lower_bound, upper_bound)

my_interval


# 4. Simulate $1000$ times from the posterior predictive distribution. Call your samples `post_pred_samples`.

# In[18]:


num_simulations = 1000

posterior_samples = stats.beta.rvs(posterior_hyperparam1, posterior_hyperparam2, size=num_simulations)
post_pred_samples = np.random.binomial(1, posterior_samples, size=num_simulations)

post_pred_samples


# In[ ]:




