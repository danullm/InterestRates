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

#def hw_process_spline(r0, kappa, sigma, spl, T = 5., N = 100, seed = 0):
#    if seed != 0:
#        np.random.seed(seed)
#
#    dt = T/N
#    t = 0.
#    rates = [r0]
#    for i in range(N):
#        dr = kappa*(theta_hw(spl, kappa, sigma, t)-rates[-1])*dt
#        dr += sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
#        dr = float(dr)
#        t += dt
#        rates.append(rates[-1] + dr)
#    
#    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

def theta_hw(spl, kappa, sigma, t):
    tmp = spl.derivative()(t)
    tmp += spl(t)*kappa
    tmp += sigma**2/2/kappa*(1-np.exp(-2*kappa*t))
    tmp = tmp / kappa
    return(tmp)

def hw_B(kappa, T, t=0.):
    if t > 0:
        T = T-t
    tmp = 1 - np.exp(-kappa*T)
    tmp = tmp / kappa
    return(tmp)
    
    
def hw_A(kappa, sigma, forward_spl, T, t = 0.):
    P0T = np.exp(-myfun(forward_spl, T))
    P0t = np.exp(-myfun(forward_spl, t))
    
    tmp1 = -hw_B(kappa, T, t)*forward_spl(t)
    tmp2 = sigma**2 * ( np.exp(- kappa*T ) - np.exp(- kappa*t ) )**2
    tmp2 = tmp2 * (np.exp(2*kappa*t) - 1) /4/kappa**3
    tmp = tmp1 - tmp2
    tmp = np.exp(tmp)
    tmp = P0T/P0t*tmp
    return(tmp)
    
def hw_P(kappa, sigma, forward_spl, r0, T, t = 0.):
    return( hw_A(kappa, sigma, forward_spl, T, t) * np.exp(- hw_B(kappa, T, t) * r0 ) )
    
    
def myfun(forward_spl, x):
    return(forward_spl.integral(0, x))
    
vfunc = np.vectorize(myfun)

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
    
    discounts = np.exp(-vfunc(forward_spl, xnew))

    plt.plot(xnew, discounts)

    kappa = 0.1
    sigma = 0.7 / np.sqrt(250)
    
    plt.plot(xnew, hw_P(kappa, sigma, forward_spl, r0, xnew, 0))
    

    