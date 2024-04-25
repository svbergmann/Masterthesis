#!/usr/bin/env python
# coding: utf-8

# In[2]:


from itertools import product

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import sympy as sp
from IPython.display import display
from sympy import abc, oo, Rational, init_printing

import checked_functions as c_f
import symbols as sym

init_printing()


# # This document aims to numerically check $\overline{w'r_t'\theta_l'}$

# ## Define the normal distributions.

# In[3]:


display(sp.Eq(sp.symbols('\\mu_1'), c_f.mu_1_w_theta_l_r_t, evaluate=False))
display(sp.Eq(sp.symbols('\\Sigma_1'), c_f.Sigma_1_w_theta_l_r_t, evaluate=False))
#display(sp.Eq(sym.G_1_w_theta_l_r_t, c_f.G_1_w_theta_l_r_t_density))


# In[4]:


display(sp.Eq(sp.symbols('\\mu_2'), c_f.mu_2_w_theta_l_r_t, evaluate=False))
display(sp.Eq(sp.symbols('\\Sigma_2'), c_f.Sigma_2_w_theta_l_r_t, evaluate=False))
#display(sp.Eq(sym.G_2_w_theta_l_r_t, c_f.G_2_w_theta_l_r_t_density))


# In[5]:


display(sp.Eq(sp.symbols('\\mu_3'), c_f.mu_3_w_theta_l_r_t, evaluate=False))
display(sp.Eq(sp.symbols('\\Sigma_3'), c_f.Sigma_3_w_theta_l_r_t, evaluate=False))
#display(sp.Eq(sym.G_3_w_theta_l_r_t, c_f.G_3_w_theta_l_r_t_density))


# In[6]:


display(sp.Eq(sym.G_w_theta_l_r_t, c_f.G_w_theta_l_r_t))


# In[7]:


w_prime_r_t_prime_theta_l_prime_bar = sp.Integral(
    (sp.abc.w - sym.w_bar) * (sym.r_t - sym.r_t_bar) * (sp.abc.theta - sym.theta_l_bar) * c_f.G_w_theta_l_r_t,
    [sp.abc.w, -oo, oo],
    [sym.theta_l, -oo, oo],
    [sym.r_t, -oo, oo])
display(sp.Eq(sym.w_prime_r_t_prime_theta_l_prime_bar, w_prime_r_t_prime_theta_l_prime_bar))


# In[8]:


w_prime_r_t_prime_theta_l_prime_bar = (
    w_prime_r_t_prime_theta_l_prime_bar.subs({
        sym.w_bar: c_f.w_bar(),
        sym.r_t_bar: c_f.r_t_bar(),
        sym.theta_l_bar: c_f.theta_l_bar()
    }))
display(sp.Eq(sym.w_prime_r_t_prime_theta_l_prime_bar, w_prime_r_t_prime_theta_l_prime_bar))


# The equation in the document is:

# In[9]:


display(sp.Eq(sym.w_prime_r_t_prime_theta_l_prime_bar, c_f.w_prime_r_t_prime_theta_l_prime_bar()))


# Since the integral is too difficult to be calculated analytically, at least with sympy, we try to put in some arbitrary numbers for the pdf parameters, to simplify the equations.

# We also use the method `nquad(..)` from `sympy` to get a numerical evaluation of the 3d integral.

# We create a dataframe to get all possible permutations and therefore also all possible evaluations of the integrals.

# In[10]:


df = pd.DataFrame(
    product([3, 1],
            [-2],
            [-1, 3],
            [2],
            [1, 4],
            [2],
            [1.1],
            [1.3],
            [1.4],
            [1.7],
            [1.2],
            [1.5],
            [1.9],
            [1.6],
            [.55],
            [.8],
            [.65],
            [.45],
            [.35],
            [.5]),
    columns=[sym.w_1,
             sym.w_2,
             sym.theta_l_1,
             sym.theta_l_2,
             sym.r_t_1,
             sym.r_t_2,
             sym.sigma_theta_l_1,
             sym.sigma_theta_l_2,
             sym.sigma_theta_l_3,
             sym.sigma_w,
             sym.sigma_r_t_1,
             sym.sigma_r_t_2,
             sym.sigma_r_t_3,
             sym.sigma_w_3,
             sp.abc.alpha,
             sp.abc.delta,
             sym.rho_w_theta_l,
             sym.rho_w_r_t,
             sym.rho_theta_l_r_t,
             sym.r_r_t_theta_l])


# In[11]:


w_prime_r_t_prime_theta_l_prime_bar_check_sym_val = (
    c_f.w_prime_r_t_prime_theta_l_prime_bar().subs({
        sym.w_bar: c_f.w_bar(),
        sym.r_t_bar: c_f.r_t_bar(),
        sym.theta_l_bar: c_f.theta_l_bar()
    }))


# In[12]:


df['check_val'] = (
    df.apply(lambda x: Rational(c_f.w_prime_r_t_prime_theta_l_prime_bar().subs({
        sym.w_bar: c_f.w_bar(),
        sym.theta_l_bar: c_f.theta_l_bar(),
        sym.r_t_bar: c_f.r_t_bar(),
        sym.w_1: x[sym.w_1],
        sym.w_2: x[sym.w_2],
        sym.theta_l_1: x[sym.theta_l_1],
        sym.theta_l_2: x[sym.theta_l_2],
        sym.r_t_1: x[sym.r_t_1],
        sym.r_t_2: x[sym.r_t_2],
        sym.sigma_theta_l_1: x[sym.sigma_theta_l_1],
        sym.sigma_theta_l_2: x[sym.sigma_theta_l_2],
        sym.sigma_r_t_1: x[sym.sigma_r_t_1],
        sym.sigma_r_t_2: x[sym.sigma_r_t_2],
        sp.abc.alpha: x[sp.abc.alpha],
        sp.abc.delta: x[sp.abc.delta],
        sym.r_r_t_theta_l: x[sym.r_r_t_theta_l]
    }).evalf()), axis=1))


# Calculate the moment numerically:

# In[13]:


import scipy

df['num_int'] = df.apply(lambda x: scipy.integrate.nquad(
    sp.lambdify(
        [sp.abc.w, sym.r_t, sym.theta_l],
        (((sp.abc.w - c_f.w_bar()) *
          (sym.r_t - c_f.r_t_bar()) *
          (sym.theta_l - c_f.theta_l_bar()) *
          c_f.G_w_theta_l_r_t))
        .subs({
            sym.w_bar: c_f.w_bar(),
            sym.r_t_bar: c_f.r_t_bar(),
            sym.theta_l_bar: c_f.theta_l_bar(),
            sym.w_1: x[sym.w_1],
            sym.w_2: x[sym.w_2],
            sym.theta_l_1: x[sym.theta_l_1],
            sym.theta_l_2: x[sym.theta_l_2],
            sym.r_t_1: x[sym.r_t_1],
            sym.r_t_2: x[sym.r_t_2],
            sym.sigma_w: x[sym.sigma_w],
            sym.sigma_w_3: x[sym.sigma_w_3],
            sym.sigma_theta_l_1: x[sym.sigma_theta_l_1],
            sym.sigma_theta_l_2: x[sym.sigma_theta_l_2],
            sym.sigma_theta_l_3: x[sym.sigma_theta_l_3],
            sym.sigma_r_t_1: x[sym.sigma_r_t_1],
            sym.sigma_r_t_2: x[sym.sigma_r_t_2],
            sym.sigma_r_t_3: x[sym.sigma_r_t_3],
            sp.abc.alpha: x[sp.abc.alpha],
            sp.abc.delta: x[sp.abc.delta],
            sym.rho_w_theta_l: x[sym.rho_w_theta_l],
            sym.rho_w_r_t: x[sym.rho_w_r_t],
            sym.rho_theta_l_r_t: x[sym.rho_theta_l_r_t],
            sym.r_r_t_theta_l: x[sym.r_r_t_theta_l]
        })),
    ranges=[[-30, 30], [-30, 30], [-30, 30]])[0],
                         axis=1)


# In[14]:


df['diff'] = abs(df['check_val'] - df['num_int'])


# In[15]:


df['diff_num'] = abs(df['check_val'].astype(float) - df['num_int'])


# In[16]:


display(df)


# In[17]:


import numpy as np

print('The mean error between the rhs and the lhs is:', np.mean(df['diff_num']))

