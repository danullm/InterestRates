#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:54:22 2018

@author: daniel
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import scipy.optimize as sop

def cap(T, sigma, kappa, discount, forward_rates, typ = 'black'):
    val = 0
    for i in range(len(T)-1):
        T0 = T[i]
        T1 = T[i+1]
        if typ == 'black':
            val += cpl_black(T0, T1, sigma, kappa, discount, forward_rates)
        if typ == 'normal':
            val += cpl_bachelier(T0, T1, sigma, kappa, discount, forward_rates)
    return(val)

def cpl_black(T0, T1, sigma, kappa, discount, forward_rates):
    delta = T1 - T0
    d1, d2 = cpl_black_d(T0, T1, sigma, kappa, discount, forward_rates)
    tmp = float(delta * discount.loc[T1])
    tmp *= (float(forward_rates.loc[T1])*norm.cdf(d1) - kappa*norm.cdf(d2) )
    return(tmp)


def cpl_black_d(T0, T1, sigma, kappa, discount, forward_rates):
    tmp1 = np.log(forward_rates.loc[T1]/kappa)
    tmp2 = sigma**2*(T0)
    d1, d2 = [ float((tmp1 + 0.5*tmp2)/np.sqrt(tmp2)) , float((tmp1 - 0.5*tmp2)/np.sqrt(tmp2)) ]
    return([d1, d2])


def get_discount(forward_rates):
    discount = pd.DataFrame()
    for i in range(len(forward_rates)):
        T0 = forward_rates.iloc[i,].T0
        T1 = forward_rates.iloc[i,].T1
        delta = T1 - T0
        F = forward_rates.iloc[i,].Rate
        
        if T0 == 0:
            P_tmp = pd.DataFrame(data = [1/(F*delta+1)], index = [T1])
        else:
            P_tmp = pd.DataFrame(data = [float(discount.loc[T0])/(F*delta+1)], index = [T1])
        
        discount = discount.append(P_tmp)
        
    return(discount)

def my_func(sigma):
    global kappa, T, discount, forward_rates, price
    
    return(cap(T, sigma, kappa, discount, forward_rates) - price)**2
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

rates = [6,8,9,10,10,10,9,9]
rates = [x/100 for x in rates]

T = np.arange(0.25,2.25,0.25)

forward_rates = pd.DataFrame({'T0':T-0.25, 'T1':T, 'Rate':rates})

discount = get_discount(forward_rates)


forward_rates = pd.DataFrame(data = rates, index = T)

cap_maturity = [2]
cap_price = [1]
cap_price = [x/100 for x in cap_price]

cap_rate = (discount.loc[0.25] - discount.loc[2])/0.25/(discount.sum()-discount.loc[0.25])

price = 0.01

cap(T, 0.05, cap_rate, discount, forward_rates)

sop.minimize(my_func, 0.05).x*100
