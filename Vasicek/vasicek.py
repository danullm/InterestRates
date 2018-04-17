#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:36:13 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
sys.path.insert(0, '/home/daniel/Seafile/Dani/Python/InterestRates/Svensson/')

from svensson import *

#------------------------------------------------------------------------------

def vasicek_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        dr = float(dr)
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

#------------------------------------------------------------------------------

def vasicek_mean(r0, theta, kappa, sigma, T = 1., N = 100):
    t = range(N+1)
    dt = T/N
    t = [x*dt for x in t]
    y = [np.exp(-kappa*x)*r0+theta*(1-np.exp(-kappa*x)) for x in t]
    y = np.array(y)
    return(pd.DataFrame(data = y, index = t))

#------------------------------------------------------------------------------

def vasicek_sd(r0, theta, kappa, sigma, T = 1., N = 100, alpha = 0.90):
    dt = T/N
    alpha = 1-alpha
    t = range(N+1)
    t = [x*dt for x in t]
 
    y = np.sqrt(np.array([sigma**2/2/kappa*(1-np.exp(-2*kappa*x)) for x in t]))

    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower = pd.DataFrame(data = y*norm.cdf(1-alpha/2), index = means.index)
    upper = means + lower
    lower = means - lower
        
    return(lower, upper)

#------------------------------------------------------------------------------

def vasicek_B(kappa, T, t = 0.):
    if t > 0:
        T = T-t
    return( (1-np.exp(-kappa*T))/kappa )

def vasicek_B_prime(kappa, T):
    return(1 - kappa*vasicek_B(kappa, T))

def vasicek_B_2prime(kappa, T):
    return(-kappa + kappa**2*vasicek_B(kappa, T))


#------------------------------------------------------------------------------
    
def vasicek_A(theta, kappa, sigma, T, t = 0.):
    if t > 0:
        T = T-t
    tmp = -(theta-sigma**2/2/kappa**2)*(vasicek_B(kappa, T, t) - T + t)
    tmp += sigma**2/4/kappa * vasicek_B(kappa, T, t)**2
    return(tmp)

def vasicek_A_prime(kappa, theta, sigma, T):
    return(kappa*theta*vasicek_B(kappa, T) - sigma**2/2*vasicek_B(kappa, T))

def vasicek_A_2prime(kappa, theta, sigma, T):
    tmp = kappa*theta*vasicek_B_prime(kappa, T)
    tmp -= sigma**2*vasicek_B(kappa, T)*vasicek_B_prime(kappa, T)
    return(tmp)

#------------------------------------------------------------------------------

def vasicek_discount_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    t = np.linspace(0,T,N)
    A = np.array([vasicek_A(theta, kappa, sigma, T) for T in t])
    B = np.array([vasicek_B(kappa, T) for T in t])
    tmp = np.exp(-A-B*r0)
    tmp = pd.DataFrame(data = tmp, index = t)
    return(tmp)

def vasicek_discount_factor(r0, theta, kappa, sigma, T, t):
    return(np.exp( -vasicek_A(theta, kappa, sigma, T, t) - vasicek_B(kappa, T, t)*r0 ))

#------------------------------------------------------------------------------

def vasicek_yield_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    discount_curve = vasicek_discount_curve(r0, theta, kappa, sigma, T, N)
    t = discount_curve.index
    y = [r0]
    for x in t[1:]:
        y.append(-np.log(discount_curve.loc[x,:].values[0])/x)
    tmp = pd.DataFrame(data = y, index = t)    
    return(tmp)
    
#def vasicek_forward_curve(r0, theta, kappa, sigma, T = 10, N = 50):
#    t = np.linspace(0,T,N)
#    A = vasicek_A(theta, kappa, sigma, t)
#    B = vasicek_B(kappa, t)
#    
#    tmp = kappa*theta*B - sigma**2/2*B**2
#    tmp += (1 - kappa*B)*r0
#    
#    tmp = pd.DataFrame(data = tmp, index = t)
#    
#    return(tmp)

def vasicek_forward_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    t = np.linspace(0,T,N)
    tmp = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t) * r0
    tmp = pd.DataFrame(data = tmp, index = t)
    return(tmp)

    
#------------------------------------------------------------------------------

if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    
    T , N, mc = [25., 500, 5]
    
    plot = False
    
    rates = vasicek_process(r0, theta, kappa, sigma, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, vasicek_process(r0, theta, kappa, sigma, T, N)], axis = 1)
    
    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower, upper = vasicek_sd(r0, theta, kappa, sigma, T, N, alpha = .99)
    
    DFC = vasicek_discount_curve(r0, theta, kappa, sigma, T)    
    YC = vasicek_yield_curve(r0, theta, kappa, sigma, T)
    FC = vasicek_forward_curve(r0, theta, kappa, sigma, T)
    
    if plot == True:
        
        fig, axes = plt.subplots(3,1, sharex = True)
        
        axes[0].plot(rates, alpha = 0.25)
        axes[0].plot(means, c = 'red')
        axes[0].plot(lower, c = 'red', linestyle = '--')
        axes[0].plot(upper, c = 'red', linestyle = '--')
        axes[0].axhline(theta, c = 'black', linestyle = ':')
        
        axes[1].plot(DFC, label = 'Discount Curve')
        axes[1].legend()
        
        axes[2].plot(YC, label = 'Yield Curve')
        axes[2].plot(FC, label = 'Forward Curve')
        axes[2].legend()
        
        [x.grid() for x in axes]
        fig.tight_layout()
        plt.show()
        
        
    rate = rates.iloc[:,0]
    rate.plot()
    plt.plot(np.exp(-rate.cumsum()*rate.index[1]))