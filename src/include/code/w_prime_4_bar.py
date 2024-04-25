#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sp
from IPython.display import display
from sympy import abc, oo, init_printing

import checked_functions as c_f
import symbols as sym
init_printing()


# # This document aims to analytically check $\overline{w'^4}$

# ## Define the marginal distributions with those parameters.

# In[2]:


display(sp.Eq(sym.G_1_w, c_f.G_1_theta_l_density))


# In[3]:


display(sp.Eq(sym.G_2_w, c_f.G_2_theta_l_density))


# In[4]:


display(sp.Eq(sym.G_3_w, c_f.G_3_theta_l_density))


# In[5]:


display(sp.Eq(sym.G_w, c_f.G_w))


# Calculate the moment analytically:

# In[6]:


w_prime_4_bar_int = sp.Integral((sp.abc.w - sym.w_bar) ** 4 * c_f.G_w, [sp.abc.w, -oo, oo])
display(sp.Eq(sym.w_prime_4_bar, w_prime_4_bar_int))


# In[7]:


w_prime_4_bar_int_val = w_prime_4_bar_int.doit(conds='none').simplify()
display(sp.Eq(sym.w_prime_4_bar, w_prime_4_bar_int_val))


# The equation in the document is:

# In[8]:


display(sp.Eq(sym.w_prime_4_bar, c_f.w_prime_4_bar()))


# where

# In[9]:


display(sp.Eq(sym.sigma_tilde_w, c_f.sigma_tilde_w()))


# and

# In[10]:


display(sp.Eq(sym.w_prime_2_bar, c_f.w_prime_2_bar()))


# and

# In[11]:


display(sp.Eq(sym.w_prime_3_bar, c_f.w_prime_2_bar()))


# So,

# In[12]:


lambda_w_val = c_f.lambda_w().subs({
    sym.w_prime_2_bar: c_f.w_prime_2_bar()
})
display(sp.Eq(sym.lambda_w, lambda_w_val))


# In[13]:


w_prime_4_bar_check_val = c_f.w_prime_4_bar().subs({
    sym.w_prime_2_bar: c_f.w_prime_2_bar(),
    sym.w_prime_3_bar: c_f.w_prime_3_bar(),
    sym.sigma_tilde_w: c_f.sigma_tilde_w().subs({
        sym.w_prime_2_bar: c_f.w_prime_2_bar(),
        sym.lambda_w: c_f.lambda_w()
    })
})

display(sp.Eq(sym.w_prime_4_bar, w_prime_4_bar_check_val))


# In[14]:


display(sp.Eq(w_prime_4_bar_int_val, w_prime_4_bar_check_val, evaluate=True)
        .subs({sym.w_bar: c_f.w_bar()}).simplify())

