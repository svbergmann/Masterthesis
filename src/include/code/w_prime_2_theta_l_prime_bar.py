#!/usr/bin/env python
# coding: utf-8

# In[17]:


from itertools import product

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import sympy as sp
from IPython.display import display
from sympy import abc, oo, Rational

import checked_functions as c_f
import symbols as sym


# # This document aims to numerically check $\overline{w'^2\theta_l'}$

# ## Define the marginal distributions.

# In[18]:


display(sp.Eq(sym.G_1_w_theta, c_f.G_1_w_theta_density))


# In[19]:


display(sp.Eq(sym.G_2_w_theta, c_f.G_2_w_theta_density))


# In[20]:


display(sp.Eq(sym.G_3_w_theta, c_f.G_3_w_theta_density))


# In[21]:


display(sp.Eq(sym.G_w_theta, c_f.G_w_theta, evaluate=False))


# In[22]:


w_prime_2_theta_l_prime_bar = sp.Integral(
    (sp.abc.w - sym.w_bar) ** 2 * (sp.abc.theta - sym.theta_l_bar) * c_f.G_w_theta,
    [sp.abc.w, -oo, oo],
    [sp.abc.theta, -oo, oo])
display(sp.Eq(sym.w_prime_2_theta_prime_l_bar, w_prime_2_theta_l_prime_bar))


# In[23]:


w_prime_2_theta_l_prime_bar = (
    w_prime_2_theta_l_prime_bar.subs({
        sym.w_bar: c_f.w_bar(),
        sym.theta_l_bar: c_f.theta_l_bar()
    }))

display(sp.Eq(sym.w_prime_2_theta_prime_l_bar,
              w_prime_2_theta_l_prime_bar))


# The equation in the document is:

# In[24]:


display(sp.Eq(sym.w_prime_2_theta_prime_l_bar, c_f.w_prime_2_theta_l_prime_bar()))


# For this we still need $\tilde{\sigma_w}$:

# In[25]:


display(sp.Eq(sym.sigma_tilde_w, c_f.sigma_tilde_w()))


# And $\overline{w'^{2}}$:

# In[26]:


display(sp.Eq(sym.w_prime_2_bar, c_f.w_prime_2_bar()))


# And $\overline{w'^{3}}$:

# In[27]:


display(sp.Eq(sym.w_prime_3_bar, c_f.w_prime_3_bar()))


# And $\lambda_w$:

# In[28]:


display(sp.Eq(sym.lambda_w, c_f.lambda_w()))


# And $\lambda_{w\theta}$:

# In[29]:


display(sp.Eq(sym.lambda_w_theta, c_f.lambda_w_theta()))


# And $\overline{w'\theta_l'}$:

# In[30]:


display(sp.Eq(sym.w_prime_theta_l_prime_bar, c_f.w_prime_theta_l_prime_bar()))


# Putting those all together yields:

# In[31]:


w_prime_2_theta_l_prime_bar_check_val = (
    c_f.w_prime_2_theta_l_prime_bar().subs({
        sym.sigma_tilde_w: c_f.sigma_tilde_w(),
        sym.lambda_w: c_f.lambda_w(),
        sym.w_prime_2_bar: c_f.w_prime_2_bar(),
        sym.w_prime_3_bar: c_f.w_prime_3_bar(),
        sym.lambda_w_theta: c_f.lambda_w_theta(),
        sym.w_prime_theta_l_prime_bar: c_f.w_prime_theta_l_prime_bar()
    }))

w_prime_2_theta_l_prime_bar_check_val = (
    w_prime_2_theta_l_prime_bar_check_val.subs({
        sym.w_prime_2_bar: c_f.w_prime_2_bar(),
        sym.w_bar: c_f.w_bar(),
        sym.theta_l_bar: c_f.theta_l_bar()
    }))

display(sp.Eq(sym.w_prime_2_theta_prime_l_bar,
              w_prime_2_theta_l_prime_bar_check_val))


# Since the integral is too difficult to be calculated analytically, at least with sympy, we try to put in some arbitrary numbers for the pdf parameters, to simplify the equations.

# We create a dataframe to get all possible permutations and therefore also all possible evaluations of the integrals.

# In[32]:


df = pd.DataFrame(
    product([0, 1],
            [-2, 2],
            [-1, 2],
            [0, 3],
            [Rational(1, 10)],
            [Rational(3, 10)],
            [Rational(4, 10)],
            [Rational(7, 10)],
            [Rational(6, 10)],
            [Rational(5, 10)],
            [Rational(1, 10), Rational(5, 10)],
            [Rational(5, 10)]),
    columns=[sym.w_1,
             sym.w_2,
             sym.theta_l_1,
             sym.theta_l_2,
             sym.sigma_theta_l_1,
             sym.sigma_theta_l_2,
             sym.sigma_theta_l_3,
             sym.sigma_w,
             sym.sigma_w_3,
             sp.abc.alpha,
             sp.abc.delta,
             sym.rho_w_theta_l])


# In[33]:


df['check_val'] = (
    df.apply(lambda x:
             w_prime_2_theta_l_prime_bar_check_val.subs({
                 sym.w_1: x[sym.w_1],
                 sym.w_2: x[sym.w_2],
                 sym.theta_l_1: x[sym.theta_l_1],
                 sym.theta_l_2: x[sym.theta_l_2],
                 sym.sigma_theta_l_1: x[sym.sigma_theta_l_1],
                 sym.sigma_theta_l_2: x[sym.sigma_theta_l_2],
                 sym.sigma_theta_l_3: x[sym.sigma_theta_l_3],
                 sym.sigma_w: x[sym.sigma_w],
                 sym.sigma_w_3: x[sym.sigma_w_3],
                 sp.abc.alpha: x[sp.abc.alpha],
                 sp.abc.delta: x[sp.abc.delta],
                 sym.rho_w_theta_l: x[sym.rho_w_theta_l]
             }), axis=1))


# Calculate the moment analytically:

# In[34]:


df['num_int'] = (
    df.apply(lambda x: Rational(w_prime_2_theta_l_prime_bar.subs({
        sym.w_1: x[sym.w_1],
        sym.w_2: x[sym.w_2],
        sym.theta_l_1: x[sym.theta_l_1],
        sym.theta_l_2: x[sym.theta_l_2],
        sym.sigma_theta_l_1: x[sym.sigma_theta_l_1],
        sym.sigma_theta_l_2: x[sym.sigma_theta_l_2],
        sym.sigma_theta_l_3: x[sym.sigma_theta_l_3],
        sym.sigma_w: x[sym.sigma_w],
        sym.sigma_w_3: x[sym.sigma_w_3],
        sp.abc.alpha: x[sp.abc.alpha],
        sp.abc.delta: x[sp.abc.delta],
        sym.rho_w_theta_l: x[sym.rho_w_theta_l]
    }).doit(conds='none', method='quad').evalf()), axis=1))


# In[35]:


df['diff'] = abs(df['check_val'] - df['num_int'])


# In[36]:


df['diff_num'] = abs(df['check_val'].astype(float) - df['num_int'].astype(float))


# In[37]:


display(df)


# In[38]:


import numpy as np

print('The mean error between the rhs and the lhs is:', np.mean(df['diff_num']))

