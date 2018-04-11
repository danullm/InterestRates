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

def vasicek_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

def vasicek_mean(r0, theta, kappa, sigma, T = 1., N = 100):
    t = range(N+1)
    dt = T/N
    t = [x*dt for x in t]
    y = [np.exp(-kappa*x)*r0+theta*(1-np.exp(-kappa*x)) for x in t]
    y = np.array(y)
    return(pd.DataFrame(data = y, index = t))

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

def vasicek_B(kappa, T, t = 0):
    if t > 0:
        T = T-t
    return( (1-np.exp(-kappa*T))/kappa )
    
def vasicek_A(theta, kappa, sigma, T, t = 0):
    if t > 0:
        T = T-t
    
    tmp = (theta-sigma**2/2/kappa**2)*(vasicek_B(kappa, T, t) - T + t)
    tmp -= sigma**2/4/kappa * vasicek_B(kappa, T, t)**2

    

if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    
    T , N, mc = [10., 200, 5]
    
    plot = False
    
    rates = vasicek_process(r0, theta, kappa, sigma, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, vasicek_process(r0, theta, kappa, sigma, T, N)], axis = 1)
    
    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower, upper = vasicek_sd(r0, theta, kappa, sigma, T, N, alpha = .99)
    
    if plot == True:
    
        plt.plot(rates, alpha = 0.25)
        plt.plot(means, c = 'red')
        plt.plot(lower, c = 'red', linestyle = '--')
        plt.plot(upper, c = 'red', linestyle = '--')
        plt.axhline(theta, c = 'black', linestyle = ':')
        plt.grid()
        plt.tight_layout()
