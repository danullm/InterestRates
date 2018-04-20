#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:34:07 2018

@author: daniel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as sop

def bond_vola_integral(beta, vola, times):
    tmp = vola**2/beta**2 
    tmp *= ( np.exp( -beta*times[0] ) - np.exp( -beta*times[1] ) )**2
    tmp *= ( np.exp( 2*beta*times[0] ) - 1 )/2/beta
    return(tmp)
    
    
def cpl_Gauss_HJM(betas, volas, times, k, discount):
    T1 = times[1]
    T0 = times[0]
    delta = T1-T0 
    d1, d2 = cpl_Gauss_HJM_d(betas, volas, times, k, discount)
    tmp = 1+delta*k
    tmp2 = discount.loc[T0]*norm.cdf(-d2) - tmp * discount.loc[T1]*norm.cdf(-d1)
    return(tmp2)


def cpl_Gauss_HJM_d(beta, vola, times, k, discount):
    T1 = times[1]
    T0 = times[0]
    delta = T1-T0
    tmp1 = np.log(discount.loc[T1]/discount.loc[T0]*(1+delta*k))
    tmp2 = bond_vola_integral(beta, vola, times)
    return([(tmp1+0.5*tmp2)/np.sqrt(tmp2), (tmp1-0.5*tmp2)/np.sqrt(tmp2)])


def cp_Gauss_HJM(T, k, betas, volas, discount):
    val = 0
    for i in range(len(T)-1):
        times = [T[i], T[i+1]]
        val += cpl_Gauss_HJM(betas, volas, times, k, discount)
    return(val)

def error_function(p0):
    
    global i, min_RMSE, discount
    
    v, beta = p0
    
    if v == 0.0 or beta == 0.0:
        return 500.0
    
    se = []
    for t, x in calibration_data.iterrows():
        k = x[2]
        fy = x[1]
        Tmax = x[0]
        T = np.arange(0.5, Tmax+0.5, 0.5)
        model_value = cp_Gauss_HJM(T, k, beta, v, discount)
        se.append((model_value - fy) ** 2)
        
    RMSE = float(np.sqrt(sum(se) / len(se)))
    min_RMSE = min(min_RMSE, RMSE)
    
    if i % 50 == 0:
       print('%4d |' % i, '%2.3f, %2.3f' % tuple(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    
    return RMSE


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
    i = 0           # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    
    rates = [6,8,9,10,10,10,9,9]
    rates = [x/100 for x in rates]
    T0 = np.arange(0,4,0.5)
    T1 = np.arange(0.5,4.5,0.5)
    
    forward_rates = pd.DataFrame({'T0':T0, 'T1':T1, 'Rate':rates})
    
    
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
        
    
    
    cap_maturity = [1,2,3,4]
    cap_price = [.2, .8, 1.2, 1.6]
    cap_price = [x/100 for x in cap_price]
    cap_strikes = forward_rates[forward_rates.T1.isin(cap_maturity)].Rate
    
    calibration_data = pd.DataFrame({'Mat': cap_maturity, 'Price': cap_price, 'Stike': cap_strikes})
    
    p0 = sop.brute(error_function,
               ((0.02, 0.05, 0.005),  #v,
                (0.1, 2., 0.1)),   #beta
                finish=None)
               
    opt = sop.fmin(error_function, p0, maxiter=5000, 
               maxfun=750, xtol=0.000001, ftol=0.000001)


