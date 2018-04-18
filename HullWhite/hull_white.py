#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:22:45 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def forward_curve(xnew, discount, tenor = 1):
    """ Input variables:
    xnew        grid to be evaluated
    discount    the discount curve as interpolated object
    tenor       tenor of the forward rate
    """
    if tenor == 0:
        tmp = discount.derivative()
        return(-tmp(xnew))
    else:
        rates = pd.DataFrame(data = [np.nan]*len(xnew), index = xnew)
        for x in xnew:
            try:
                rates.loc[x] = (discount(x)/discount(x+tenor)-1)*(1./tenor)
            except:
                rates.loc[x] = np.nan        
        return(rates[0])

def hw_process_spline(r0, kappa, sigma, spl, T = 5., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    t = 0.
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta_hw(spl, kappa, sigma, t)-rates[-1])*dt
        dr += sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        dr = float(dr)
        t += dt
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

def theta_hw(spl, kappa, sigma, t):
    tmp = spl.derivative()(t)
    tmp += spl(t)*kappa
    tmp += sigma**2/2/kappa*(1-np.exp(-2*kappa*t))
    tmp = tmp / kappa
    return(tmp)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    t = np.arange(0,11)
    f = [.005, .0075, .0135, .01755, .0202, .02136, .02285, .02372, .02482, .02561, .02608]

    forward_spl = InterpolatedUnivariateSpline(t, f)
    r0 = forward_spl(0)
    
    T = int(max(t))
    N = T*100
    mc = 10
    
    xnew = np.linspace(0, T, N)
    
    def myfun(forward_spl, x):
        return(forward_spl.integral(0, x))
    
    vfunc = np.vectorize(myfun)
    
    discounts = np.exp(-vfunc(forward_spl, xnew))

    kappa = 0.1
    sigma = 0.7 / np.sqrt(250)
    
    rates = hw_process_spline(r0, kappa, sigma, forward_spl, T, N)
    
    for i in range(mc-1):
        rates = pd.concat([rates, hw_process_spline(r0, kappa, sigma, forward_spl, T, N) ], axis = 1)
    
    plt.plot(rates)
    #plt.plot(np.exp(-rate.cumsum()*rate.index[1]))
    